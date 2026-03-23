import os
import sys
import traci
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from tqdm import tqdm

# SUMO 홈 디렉토리 환경 변수 설정
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("환경 변수 'SUMO_HOME'을 설정해주세요.")

# ==========================================
# 1. PPO 에이전트 신경망 (입력 크기 고정)
# ==========================================
class PPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOAgent, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def get_action(self, state, action_mask=None):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits = self.actor(state_tensor)
        
        if action_mask is not None:
            mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0)
            logits = logits - (1.0 - mask_tensor) * 1e9 # 불량 후보 확률 0% 강제
            
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action.item(), probs.log_prob(action), self.critic(state_tensor)

# ==========================================
# 2. SUMO 기반 현실 데이터셋(Snapshot) 수집기
# ==========================================
def collect_sumo_snapshots(num_snapshots=5000, max_svs=10):
    sumo_cmd = ["sumo", "-c", "sumo_data/sim.sumocfg", "--no-warnings"]
    traci.start(sumo_cmd)
    
    print(f"🚗 SUMO에서 {num_snapshots}개의 현실 교차로 시나리오를 추출합니다...")
    snapshots = []
    
    T_max = 2.0
    tau_guard = 0.2
    T_error = 0.5
    comm_range = 150.0
    
    step = 0
    while len(snapshots) < num_snapshots:
        traci.simulationStep()
        veh_ids = traci.vehicle.getIDList()
        
        if len(veh_ids) > 10 and step % 2 == 0:
            tv_id = np.random.choice(veh_ids)
            tv_pos = np.array(traci.vehicle.getPosition(tv_id))
            
            sv_candidates = []
            for vid in veh_ids:
                if vid != tv_id:
                    pos = np.array(traci.vehicle.getPosition(vid))
                    dist = np.linalg.norm(tv_pos - pos)
                    if dist <= comm_range:
                        sv_candidates.append(vid)
            
            # SV가 존재할 때만 스냅샷 기록 (신경망 입력을 위해 최대 max_svs 개로 자르거나 패딩)
            num_actual_svs = min(len(sv_candidates), max_svs)
            if num_actual_svs > 0:
                D_i = np.random.uniform(1.0, 5.0)
                C_i = np.random.uniform(0.5, 2.0)
                
                f_sv = np.zeros(max_svs)
                R_sv = np.zeros(max_svs)
                t_conn_actual = np.zeros(max_svs)
                t_conn_proposed = np.zeros(max_svs)
                
                # 현실적인 회전 비율(약 30%) 반영 노이즈
                is_turning = np.random.rand(num_actual_svs) < 0.3
                
                f_sv[:num_actual_svs] = np.random.uniform(1.0, 3.0, num_actual_svs)
                R_sv[:num_actual_svs] = np.random.uniform(5.0, 25.0, num_actual_svs)
                
                actual_conns = np.where(is_turning, np.random.uniform(0.5, 1.5, num_actual_svs), np.random.uniform(3.0, 6.0, num_actual_svs))
                t_conn_actual[:num_actual_svs] = actual_conns
                t_conn_proposed[:num_actual_svs] = actual_conns - np.random.uniform(0.0, 0.2, num_actual_svs)
                
                # State 구성: [D_i, C_i] + f_sv(10) + R_sv(10) + t_conn_proposed(10) -> 총 32차원
                state = np.concatenate(([D_i, C_i], f_sv, R_sv, t_conn_proposed))
                
                # Action Mask 구성 (0: Local, 1~10: SVs)
                action_mask = np.zeros(max_svs + 1, dtype=np.int8)
                action_mask[0] = 1 # Local은 항상 가능하다고 일단 가정 (안전장치)
                
                t_local = C_i / 1.0
                if t_local > T_max - tau_guard:
                    action_mask[0] = 0 # 로컬 처리도 데드라인 넘기면 마스킹
                
                for j in range(num_actual_svs):
                    t_trans = D_i / R_sv[j]
                    t_comp = C_i / f_sv[j]
                    t_total = t_trans + t_comp
                    if (t_total <= T_max - tau_guard) and (t_total <= t_conn_proposed[j] - T_error):
                        action_mask[j+1] = 1
                
                # 만약 모두 마스킹되었다면 강제로 로컬(0) 활성화하여 에러 방지
                if np.sum(action_mask) == 0: action_mask[0] = 1
                
                snapshot = {
                    'state': state,
                    'action_mask': action_mask,
                    'D_i': D_i, 'C_i': C_i, 'f_sv': f_sv, 'R_sv': R_sv,
                    't_conn_actual': t_conn_actual,
                    'num_actual_svs': num_actual_svs
                }
                snapshots.append(snapshot)
        step += 1
    traci.close()
    return snapshots

# ==========================================
# 3. 오프라인 학습 로직 (가장 공정한 비교)
# ==========================================
def train_on_snapshots(snapshots, use_masking):
    state_dim = 32 # 2 + 10 + 10 + 10
    action_dim = 11 # Local(1) + SVs(10)
    
    agent = PPOAgent(state_dim, action_dim)
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    costs_history = []
    
    T_max = 2.0
    
    print(f"🚀 PPO 학습 시작... (Masking 적용 여부: {use_masking})")
    for snap in tqdm(snapshots):
        state = snap['state']
        action_mask = snap['action_mask'] if use_masking else None
        
        action, log_prob, value = agent.get_action(state, action_mask)
        
        D_i = snap['D_i']; C_i = snap['C_i']
        f_sv = snap['f_sv']; R_sv = snap['R_sv']
        t_conn_actual = snap['t_conn_actual']
        
        E_local_base = (1.0 ** 2) * C_i
        is_failed = False
        penalty = 0.0
        
        # 선택한 액션에 대한 환경 평가
        if action == 0: # Local
            t_total = C_i / 1.0
            cost = 0.9 * (t_total / T_max) + 0.1 * 1.0
            if t_total > T_max: is_failed = True
        else: # SV Offloading
            sv_idx = action - 1
            if sv_idx >= snap['num_actual_svs']: # 패딩된 유령 차량을 고른 경우
                is_failed = True
                cost = 5.0
            else:
                t_trans = D_i / R_sv[sv_idx]
                t_comp = C_i / f_sv[sv_idx]
                t_total = t_trans + t_comp
                
                cost = 0.9 * (t_total / T_max) + 0.1 * (((0.5 * t_trans) + (f_sv[sv_idx]**2 * C_i) + 0.1) / E_local_base)
                
                if t_total > T_max: is_failed = True
                if t_total > t_conn_actual[sv_idx]: is_failed = True
        
        record_cost = 5.0 if is_failed else cost
        reward = -record_cost
        
        value_target = torch.FloatTensor([reward])
        advantage = value_target - value.squeeze()
        
        critic_loss = advantage.pow(2)
        actor_loss = -log_prob * advantage.detach()
        loss = actor_loss + 0.5 * critic_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        costs_history.append(record_cost)
        
    return costs_history

# ==========================================
# 4. 실행 및 시각화
# ==========================================
if __name__ == "__main__":
    snapshots = collect_sumo_snapshots(num_snapshots=5000)
    
    costs_proposed = train_on_snapshots(snapshots, use_masking=True)
    costs_baseline = train_on_snapshots(snapshots, use_masking=False)
    
    def moving_average(data, window_size=200):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    smooth_proposed = moving_average(costs_proposed)
    smooth_baseline = moving_average(costs_baseline)
    
    plt.figure(figsize=(10, 6))
    plt.plot(smooth_baseline, label='Standard PPO (No Filtering)', color='blue', alpha=0.5)
    plt.plot(smooth_proposed, label='Proposed Masked PPO (Stage 1 Filtering)', color='red', linewidth=2.5)
    
    plt.title('Experiment 1: Convergence of Total Cost in SUMO Environment', fontsize=16)
    plt.xlabel('Training Steps (Episodes)', fontsize=12)
    plt.ylabel('Moving Average Total Cost (Lower is Better)', fontsize=12)
    
    plt.ylim(0, 5.5)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig('exp1_sumo_convergence.png', dpi=300, bbox_inches='tight')
    print("\n✅ SUMO 데이터 기반 실험 1 완료! 'exp1_sumo_convergence.png' 파일이 저장되었습니다.")
    plt.show()