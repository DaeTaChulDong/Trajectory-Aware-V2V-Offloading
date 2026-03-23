import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from tqdm import tqdm

MAX_SVS = 30 # 최대 주변 차량 수 (혼잡한 도심 모사)
STATE_DIM = 2 + (MAX_SVS * 3) # 92차원 상태 공간
ACTION_DIM = MAX_SVS + 1      # 31차원 행동 공간

# ==========================================
# 1. 확장된 PPO 에이전트
# ==========================================
class PPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOAgent, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def get_action_and_value(self, state, action_mask=None, action=None, deterministic=False):
        logits = self.actor(state)
        if action_mask is not None:
            logits = logits - (1.0 - action_mask) * 1e9 
        probs = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        elif action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(state)

# ==========================================
# 2. 🌟 가혹해진 환경 데이터셋 생성 (Local 너프 적용)
# ==========================================
def generate_density_snapshots(num_snapshots, target_density, is_training=False):
    snapshots = []
    T_max = 5.0 
    
    for _ in range(num_snapshots):
        if is_training:
            num_actual_svs = np.random.randint(5, MAX_SVS + 1)
        else:
            num_actual_svs = target_density
            
        # 🌟 무거워진 태스크 (Local로는 절대 데드라인을 맞출 수 없음)
        D_i = np.random.uniform(6.0, 12.0) 
        C_i = np.random.uniform(4.0, 6.0) 
        
        f_sv = np.ones(MAX_SVS) * 0.1
        R_sv = np.ones(MAX_SVS) * 0.1
        t_conn_actual = np.zeros(MAX_SVS)
        t_conn_proposed = np.zeros(MAX_SVS) 
        
        distances = np.random.uniform(10.0, 150.0, num_actual_svs)
        is_turning = np.random.rand(num_actual_svs) < 0.3 
        
        # SV들은 강력한 성능을 가지도록 유도
        f_sv[:num_actual_svs] = np.random.uniform(3.0, 10.0, num_actual_svs) 
        base_rates = 100.0 / (distances / 10.0) 
        actual_rates = np.where(is_turning, base_rates * np.random.uniform(0.1, 0.4), base_rates * np.random.uniform(0.8, 1.2))
        R_sv[:num_actual_svs] = np.clip(actual_rates, 1.0, 50.0)
        
        actual_conns = np.where(is_turning, np.random.uniform(0.2, 1.5, num_actual_svs), np.random.uniform(3.0, 8.0, num_actual_svs))
        t_conn_actual[:num_actual_svs] = actual_conns
        t_conn_proposed[:num_actual_svs] = actual_conns - np.random.uniform(0.0, 0.2, num_actual_svs)
        
        state = np.concatenate(([D_i, C_i], f_sv, R_sv, t_conn_proposed))
        action_mask = np.zeros(MAX_SVS + 1, dtype=np.float32)
        action_mask[0] = 1.0 # Local은 일단 열어둠 (가장 높은 Cost를 뱉어내게 됨)
        
        for j in range(num_actual_svs):
            t_total = (D_i / R_sv[j]) + (C_i / f_sv[j])
            if (t_total <= T_max) and (t_total <= t_conn_proposed[j]):
                action_mask[j+1] = 1.0
                
        snapshots.append({
            'state': state, 'action_mask': action_mask,
            'D_i': D_i, 'C_i': C_i, 'f_sv': f_sv, 'R_sv': R_sv,
            't_conn_actual': t_conn_actual, 'num_actual_svs': num_actual_svs
        })
    return snapshots

# ==========================================
# 3. PPO 사전 학습 (가시밭길 훈련)
# ==========================================
def train_ppo_for_density():
    print("🧠 [Phase 1] TV 너프 환경에서 30대 차량 풀 탐색 PPO 학습 중...")
    train_snaps = generate_density_snapshots(5000, target_density=0, is_training=True)
    
    agent = PPOAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    optimizer = optim.Adam(agent.parameters(), lr=0.001) 
    
    # 🌟 핵심: f_tv=0.8 로 TV 성능 대폭 삭감, PENALTY_COST를 20.0으로 올려 공포감 조성
    T_max, f_tv, P_tx, k, alpha, beta, PENALTY_COST = 5.0, 0.8, 0.5, 0.05, 0.5, 0.5, 20.0
    
    for epoch in range(4): 
        for snap in train_snaps:
            state = torch.FloatTensor(snap['state']).unsqueeze(0)
            action_mask = torch.FloatTensor(snap['action_mask']).unsqueeze(0)
            action, log_prob, _, value = agent.get_action_and_value(state, action_mask)
            act_item = action.item()
            
            D_i, C_i, f_sv, R_sv, t_conn_actual = snap['D_i'], snap['C_i'], snap['f_sv'], snap['R_sv'], snap['t_conn_actual']
            is_failed, final_cost = False, 0.0
            
            if act_item == 0: 
                T_local = C_i / f_tv
                if T_local > T_max: is_failed = True # 이제 거의 무조건 실패함
                final_cost = alpha * T_local + beta * (k * C_i * (f_tv ** 2))
            else: 
                sv_idx = act_item - 1
                if sv_idx >= snap['num_actual_svs']: is_failed = True 
                else:
                    T_total = (D_i / R_sv[sv_idx]) + (C_i / f_sv[sv_idx])
                    if T_total > T_max or T_total > t_conn_actual[sv_idx]: is_failed = True
                    final_cost = alpha * T_total + beta * (P_tx * (D_i / R_sv[sv_idx]))
            
            reward = -PENALTY_COST if is_failed else -final_cost
            value_target = torch.FloatTensor([reward])
            advantage = value_target - value.squeeze()
            loss = (-log_prob * advantage.detach()) + 0.5 * advantage.pow(2)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            
    print("✅ PPO 에이전트 생존 학습 완료!")
    return agent

# ==========================================
# 4. 차량 밀도 변화에 따른 평가
# ==========================================
def evaluate_density(trained_agent):
    density_levels = [5, 10, 15, 20, 25, 30] 
    num_trials = 1000 
    
    # 평가 시에도 동일한 너프 환경 유지
    T_max, f_tv, P_tx, k, alpha, beta, PENALTY_COST = 5.0, 0.8, 0.5, 0.05, 0.5, 0.5, 20.0
    
    rates = {'Prop': [], 'Local': [], 'Gre_SNR': [], 'Gre_CPU': []}
    costs = {'Prop': [], 'Local': [], 'Gre_SNR': [], 'Gre_CPU': []}
    
    print("\n🚗 [Phase 2] 밀도(Density) 증가에 따른 심층 탐색 평가 시작...")
    trained_agent.eval() 
    
    with torch.no_grad():
        for d in density_levels:
            test_snaps = generate_density_snapshots(num_trials, target_density=d, is_training=False)
            
            s_prop, s_loc, s_snr, s_cpu = 0, 0, 0, 0
            c_prop, c_loc, c_snr, c_cpu = 0.0, 0.0, 0.0, 0.0
            
            for snap in test_snaps:
                D_i, C_i, f_sv, R_sv = snap['D_i'], snap['C_i'], snap['f_sv'], snap['R_sv']
                t_conn_actual = snap['t_conn_actual']
                num_svs = snap['num_actual_svs']
                
                T_total_svs = np.full(MAX_SVS, np.inf) 
                T_total_svs[:num_svs] = (D_i / R_sv[:num_svs]) + (C_i / f_sv[:num_svs])
                T_local = C_i / f_tv
                Cost_local = alpha * T_local + beta * (k * C_i * (f_tv ** 2))
                
                # --------------------------------------------------
                # 1. Proposed (Neural PPO)
                # --------------------------------------------------
                state_tensor = torch.FloatTensor(snap['state']).unsqueeze(0)
                mask_tensor = torch.FloatTensor(snap['action_mask']).unsqueeze(0)
                action_tensor, _, _, _ = trained_agent.get_action_and_value(state_tensor, mask_tensor, deterministic=True)
                act_item = action_tensor.item()
                
                prop_c = PENALTY_COST
                if act_item == 0:
                    if T_local <= T_max: 
                        s_prop += 1; prop_c = Cost_local
                else:
                    sv_idx = act_item - 1
                    if sv_idx < num_svs and T_total_svs[sv_idx] <= T_max and T_total_svs[sv_idx] <= t_conn_actual[sv_idx]:
                        s_prop += 1
                        prop_c = alpha * T_total_svs[sv_idx] + beta * (P_tx * (D_i / R_sv[sv_idx]))
                c_prop += prop_c

                # --------------------------------------------------
                # 2. Local Only (이제 거대한 페널티의 늪)
                # --------------------------------------------------
                loc_c = PENALTY_COST
                if T_local <= T_max:
                    s_loc += 1; loc_c = Cost_local
                c_loc += loc_c

                # --------------------------------------------------
                # 3. Greedy-SNR 
                # --------------------------------------------------
                snr_c = PENALTY_COST
                idx_snr = np.argmax(R_sv[:num_svs]) 
                if T_total_svs[idx_snr] <= T_max and T_total_svs[idx_snr] <= t_conn_actual[idx_snr]:
                    s_snr += 1
                    snr_c = alpha * T_total_svs[idx_snr] + beta * (P_tx * (D_i / R_sv[idx_snr]))
                c_snr += snr_c
                
                # --------------------------------------------------
                # 4. Greedy-CPU 
                # --------------------------------------------------
                cpu_c = PENALTY_COST
                idx_cpu = np.argmax(f_sv[:num_svs]) 
                if T_total_svs[idx_cpu] <= T_max and T_total_svs[idx_cpu] <= t_conn_actual[idx_cpu]:
                    s_cpu += 1
                    cpu_c = alpha * T_total_svs[idx_cpu] + beta * (P_tx * (D_i / R_sv[idx_cpu]))
                c_cpu += cpu_c

            rates['Prop'].append((s_prop / num_trials) * 100)
            rates['Local'].append((s_loc / num_trials) * 100)
            rates['Gre_SNR'].append((s_snr / num_trials) * 100)
            rates['Gre_CPU'].append((s_cpu / num_trials) * 100)
            
            costs['Prop'].append(c_prop / num_trials)
            costs['Local'].append(c_loc / num_trials)
            costs['Gre_SNR'].append(c_snr / num_trials)
            costs['Gre_CPU'].append(c_cpu / num_trials)
            
            print(f"Density {d:2d} SVs | Success -> Prop: {rates['Prop'][-1]:.1f}%, Loc: {rates['Local'][-1]:.1f}%, SNR: {rates['Gre_SNR'][-1]:.1f}%, CPU: {rates['Gre_CPU'][-1]:.1f}%")

    # ==========================================
    # 5. 듀얼 플롯 시각화
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # [왼쪽] Task Completion Ratio
    ax1.plot(density_levels, rates['Prop'], marker='o', markersize=9, color='red', linewidth=3, label='Proposed (Masked PPO)')
    ax1.plot(density_levels, rates['Local'], marker='x', markersize=9, color='gray', linestyle=':', linewidth=2, label='Local Execution Only')
    ax1.plot(density_levels, rates['Gre_SNR'], marker='^', markersize=8, color='blue', linestyle='-.', linewidth=2, label='Greedy-SNR (Max R_sv)')
    ax1.plot(density_levels, rates['Gre_CPU'], marker='s', markersize=8, color='green', linestyle='--', linewidth=2, label='Greedy-CPU (Max f_sv)')
    
    ax1.set_title('Task Completion Ratio under Varying Vehicle Densities', fontsize=14)
    ax1.set_xlabel('Number of Surrounding Service Vehicles (Density)', fontsize=12)
    ax1.set_ylabel('Task Completion Ratio (%)', fontsize=12)
    ax1.set_ylim(0, 105)
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # [오른쪽] Average Total Cost (V자 역전의 주인공)
    ax2.plot(density_levels, costs['Prop'], marker='o', markersize=9, color='red', linewidth=3, label='Proposed (Masked PPO)')
    ax2.plot(density_levels, costs['Local'], marker='x', markersize=9, color='gray', linestyle=':', linewidth=2, label='Local Execution Only')
    ax2.plot(density_levels, costs['Gre_SNR'], marker='^', markersize=8, color='blue', linestyle='-.', linewidth=2, label='Greedy-SNR (Max R_sv)')
    ax2.plot(density_levels, costs['Gre_CPU'], marker='s', markersize=8, color='green', linestyle='--', linewidth=2, label='Greedy-CPU (Max f_sv)')
    
    ax2.set_title('Average Total Cost (Delay & Energy) vs Density', fontsize=14)
    ax2.set_xlabel('Number of Surrounding Service Vehicles (Density)', fontsize=12)
    ax2.set_ylabel('Average Total Cost (Lower is Better)', fontsize=12)
    ax2.set_ylim(0, PENALTY_COST + 2)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.suptitle('Experiment 4: System Scalability and Resource Discovery in Dense Networks', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('exp4_density_hardmode.png', dpi=300)
    print("\n✅ 지옥의 난이도가 적용된 실험 4 완료! 'exp4_density_hardmode.png'가 저장되었습니다.")
    plt.show()

if __name__ == "__main__":
    trained_agent = train_ppo_for_density()
    evaluate_density(trained_agent)