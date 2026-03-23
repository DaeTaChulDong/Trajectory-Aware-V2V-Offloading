import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from tqdm import tqdm

# SUMO TraCI 라이브러리 임포트 (SUMO_HOME 환경변수 필요)
import os
import sys

# 1. SUMO_HOME 환경 변수 설정 (본인의 실제 설치 경로로 확인 필요)
# 보통 맥에서 brew로 설치했다면 아래 경로인 경우가 많네.
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = "/usr/local/opt/sumo/share/sumo" # 또는 실제 설치 경로

tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if tools not in sys.path:
    sys.path.append(tools)

import traci

# ==========================================
# 1. 현실적인 레퍼런스 파라미터 세팅 (Table 1 & 2 기반)
# ==========================================
TASK_TYPES = [
    {'name': 'Type 1 (Normal)', 'prob': 0.3, 'D': 50.0, 'C': 10.0, 'T_max': 10.0},
    {'name': 'Type 2 (Resource)', 'prob': 0.1, 'D': 200.0, 'C': 200.0, 'T_max': 15.0},
    {'name': 'Type 3 (Delay)', 'prob': 0.1, 'D': 10.0, 'C': 5.0, 'T_max': 5.0},
    {'name': 'Type 4 (Comp-heavy)', 'prob': 0.3, 'D': 100.0, 'C': 500.0, 'T_max': 20.0},
    {'name': 'Type 5 (Data-heavy)', 'prob': 0.2, 'D': 500.0, 'C': 100.0, 'T_max': 10.0}
]

MAX_SVS = 10           # 최대 탐색 SV 차량 수
F_TV = 2.5             # TV 로컬 CPU 속도 (GHz)
F_SV_MIN, F_SV_MAX = 10.0, 50.0  # SV CPU 속도 범위 (GHz)
COMM_RANGE = 100.0     # 통신 반경 (D_zone, meters)
P_TX = 2.0             # TV 송신 전력 (W)
KAPPA = 0.005          # 에너지 계수
ALPHA, BETA = 0.4, 0.6 # 시간, 에너지 가중치
PENALTY_COST = 50.0    # 실패 페널티

# ==========================================
# 2. 실전 SUMO V2X 환경 모듈 (Stage 1 통합)
# ==========================================
class SumoV2XEnv:
    def __init__(self, sumocfg_path):
        self.sumocfg_path = sumocfg_path
        self.state_dim = 2 + (MAX_SVS * 3) # [D, C] + [f_sv, R_sv, T_conn] * MAX_SVS
        self.action_dim = MAX_SVS + 1      # 0: Local, 1~N: SV Offloading
        
    def start_sumo(self):
        # 자네가 알려준 절대 경로를 직접 사용하네
        if not os.path.exists(self.sumocfg_path):
            sys.exit(f"❌ 파일을 찾을 수 없습니다: {self.sumocfg_path}")
            
        # --warnings false 대신 --no-warnings를 사용해야 에러가 안 나네
        sumo_cmd = [
            "sumo", # GUI를 보려면 "sumo-gui"로 변경
            "-c", self.sumocfg_path,
            "--no-warnings",
            "--no-step-log",
            "--quit-on-end"
        ]
        
        print(f"🚗 SUMO 시뮬레이션 시작 중... (경로: {self.sumocfg_path})")
        traci.start(sumo_cmd)
        
    def close_sumo(self):
        traci.close()

    def reset(self):
        # 시뮬레이션을 진행하며 차량이 2대 이상 나타날 때까지 대기
        veh_ids = []
        while len(veh_ids) < 2:
            traci.simulationStep()
            veh_ids = traci.vehicle.getIDList()
            
        # 1. 임의의 차량을 TV(Task Vehicle)로 선정
        self.tv_id = np.random.choice(veh_ids)
        self.tv_pos = traci.vehicle.getPosition(self.tv_id)
        
        # 2. 태스크 생성
        task_idx = np.random.choice(len(TASK_TYPES), p=[t['prob'] for t in TASK_TYPES])
        self.task = TASK_TYPES[task_idx]
        
        # 3. 주변 SV(Service Vehicles) 탐색 및 상태 추출
        self.sv_list = []
        self.f_sv = np.zeros(MAX_SVS)
        self.R_sv = np.ones(MAX_SVS) * 0.1 # 기본값(매우 느림)
        self.T_conn = np.zeros(MAX_SVS)
        
        sv_count = 0
        for vid in veh_ids:
            if vid == self.tv_id or sv_count >= MAX_SVS: continue
            
            sv_pos = traci.vehicle.getPosition(vid)
            dist = math.hypot(self.tv_pos[0] - sv_pos[0], self.tv_pos[1] - sv_pos[1])
            
            if dist <= COMM_RANGE: # I_zone 내에 존재하는 차량만 필터링
                self.sv_list.append(vid)
                self.f_sv[sv_count] = np.random.uniform(F_SV_MIN, F_SV_MAX)
                
                # 거리에 반비례하는 현실적인 전송률(R_sv) 계산 (Shannon 공식 모사)
                # 가까울수록 100Mbps에 가깝고, 멀어질수록 20Mbps로 떨어짐
                self.R_sv[sv_count] = max(20.0, 100.0 - (dist / COMM_RANGE) * 80.0)
                
                # 🌟 Stage 1 (Mobility Filtering): 임베딩 예측 모사
                # 실제 SUMO의 남은 경로와 상대 속도를 이용해 예상 연결 시간(T_conn) 계산
                tv_speed = traci.vehicle.getSpeed(self.tv_id)
                sv_speed = traci.vehicle.getSpeed(vid)
                rel_speed = abs(tv_speed - sv_speed) + 0.1
                
                # 코너를 도는지(라우트 차이) 여부를 반영한 연결 시간 산출 (임베딩 모델의 Output 역할)
                predicted_t_conn = (COMM_RANGE - dist) / rel_speed
                tv_route = traci.vehicle.getRoute(self.tv_id)
                sv_route = traci.vehicle.getRoute(vid)
                if tv_route[-1] != sv_route[-1]: # 목적지가 다르면 교차로에서 꺾일 확률 높음
                    predicted_t_conn *= 0.4 # 페널티 적용
                
                self.T_conn[sv_count] = min(predicted_t_conn, 30.0)
                sv_count += 1
                
        # 4. State 구성
        D_i, C_i = self.task['D'], self.task['C']
        state = np.zeros(self.state_dim, dtype=np.float32)
        state[0], state[1] = D_i / 100.0, C_i / 100.0 
        state[2:2+MAX_SVS] = self.f_sv / 50.0
        state[2+MAX_SVS:2+2*MAX_SVS] = self.R_sv / 100.0
        state[2+2*MAX_SVS:] = self.T_conn / 30.0
        
        # 🌟 5. STAGE 1 Action Masking (불확실성 원천 차단)
        action_mask = np.zeros(self.action_dim, dtype=np.float32)
        
        # 로컬 처리 통과 여부
        if (C_i / F_TV) <= self.task['T_max']:
            action_mask[0] = 1.0
            
        # 오프로딩 통과 여부 (데드라인 및 SUMO 기반 예상 연결 시간 비교)
        for j in range(sv_count):
            t_total = (D_i / self.R_sv[j]) + (C_i / self.f_sv[j])
            if t_total <= self.task['T_max'] and t_total <= self.T_conn[j]:
                action_mask[j+1] = 1.0
                
        if np.sum(action_mask) == 0:
            action_mask[0] = 1.0 # 강제 로컬 (페널티)
            
        return state, action_mask

    def step(self, action):
        D_i, C_i, T_max = self.task['D'], self.task['C'], self.task['T_max']
        t_trans, t_comp, e_trans, e_comp = 0.0, 0.0, 0.0, 0.0
        is_failed = False
        
        if action == 0: # Local
            t_comp = C_i / F_TV
            e_comp = KAPPA * C_i * (F_TV ** 2)
            if t_comp > T_max: is_failed = True
        else: # Offloading
            sv_idx = action - 1
            if sv_idx >= len(self.sv_list): 
                is_failed = True # 존재하지 않는 차량 선택 시
            else:
                t_trans = D_i / self.R_sv[sv_idx]
                t_comp = C_i / self.f_sv[sv_idx]
                e_trans = P_TX * t_trans
                
                # SUMO 실제 물리 제약 검증
                if (t_trans + t_comp) > T_max or (t_trans + t_comp) > self.T_conn[sv_idx]:
                    is_failed = True
                
        t_total = t_trans + t_comp
        e_total = e_trans + e_comp
        
        # Stage 2: Cost Calculation
        cost = ALPHA * t_total + BETA * e_total
        reward = -PENALTY_COST if is_failed else -cost
        
        info = {'t_trans': t_trans, 't_comp': t_comp, 'cost': cost, 'failed': is_failed, 'type': self.task['name']}
        
        # 다음 스텝을 위해 SUMO 시간 전진
        traci.simulationStep()
        return reward, info

# ==========================================
# 3. Stage 2: Masked PPO 에이전트
# ==========================================
class MaskedPPO(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(MaskedPPO, self).__init__()
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

    def get_action(self, state, action_mask, deterministic=False):
        logits = self.actor(state)
        # 마스킹 로직: 예측 기반 필터링으로 꺾일 확률 높은 차는 음의 무한대로 처리
        logits = logits - (1.0 - action_mask) * 1e9 
        probs = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = probs.sample()
        return action, probs.log_prob(action), self.critic(state)

# ==========================================
# 4. 실전 통합 훈련 루프
# ==========================================
def train_and_evaluate():
    # 로컬 환경 절대 경로로 업데이트
    sumo_cfg_path = "/Users/eunseo/Desktop/NetworkLab/2.20test/sumo_data/sim.sumocfg"
    
    env = SumoV2XEnv(sumo_cfg_path)
    env.start_sumo()
    
    agent = MaskedPPO(env.state_dim, env.action_dim)
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    
    print("🚀 실전 SUMO 연동: Stage 1(Mobility) + Stage 2(RL Resource) 통합 학습 시작...")
    
    try:
        # [Train Phase]
        for episode in tqdm(range(1000), desc="Training with SUMO"):
            state, mask = env.reset()
            state_t = torch.FloatTensor(state).unsqueeze(0)
            mask_t = torch.FloatTensor(mask).unsqueeze(0)
            
            action, log_prob, value = agent.get_action(state_t, mask_t)
            reward, _ = env.step(action.item())
            
            advantage = reward - value.squeeze().detach()
            loss = -log_prob * advantage + 0.5 * advantage.pow(2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("\n✅ SUMO 데이터 기반 학습 완료! 세부 지표 분석을 진행합니다.")
        
        # [Evaluation Phase]
        agent.eval()
        eval_results = {t['name']: {'t_trans': [], 't_comp': [], 'fails': 0, 'count': 0} for t in TASK_TYPES}
        
        with torch.no_grad():
            for _ in range(500):
                state, mask = env.reset()
                state_t = torch.FloatTensor(state).unsqueeze(0)
                mask_t = torch.FloatTensor(mask).unsqueeze(0)
                
                action, _, _ = agent.get_action(state_t, mask_t, deterministic=True)
                reward, info = env.step(action.item())
                
                t_name = info['type']
                eval_results[t_name]['count'] += 1
                if info['failed']:
                    eval_results[t_name]['fails'] += 1
                else:
                    eval_results[t_name]['t_trans'].append(info['t_trans'])
                    eval_results[t_name]['t_comp'].append(info['t_comp'])

    finally:
        env.close_sumo() # 에러가 나도 SUMO 소켓은 반드시 닫아주어야 함

    # ==========================================
    # 5. 결과 시각화 (T_trans vs T_comp)
    # ==========================================
    labels, avg_t_trans, avg_t_comp = [], [], []
    
    print("\n📊 [SUMO 실전 평가 결과 요약]")
    for t_name, data in eval_results.items():
        fails = data['fails']
        count = data['count']
        if count == 0: continue
        t_trans = np.mean(data['t_trans']) if data['t_trans'] else 0
        t_comp = np.mean(data['t_comp']) if data['t_comp'] else 0
        
        labels.append(t_name.split(' ')[1])
        avg_t_trans.append(t_trans)
        avg_t_comp.append(t_comp)
        
        print(f"[{t_name}] - 성공률: {((count-fails)/count)*100:.1f}% | 전송: {t_trans:.2f}s | 연산: {t_comp:.2f}s")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    width = 0.5
    
    p1 = ax.bar(x, avg_t_trans, width, label='Transmission Time (T_trans)', color='royalblue')
    p2 = ax.bar(x, avg_t_comp, width, bottom=avg_t_trans, label='Computation Time (T_comp)', color='darkorange')
    
    ax.set_ylabel('Execution Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_title('Real SUMO Data: Time Analysis by Task Type', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i in range(len(labels)):
        total_val = avg_t_trans[i] + avg_t_comp[i]
        ax.text(x[i], total_val + 0.5, f'{total_val:.1f}s', ha='center', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig('sumo_real_time_analysis.png', dpi=300)
    print("\n✅ 'sumo_real_time_analysis.png'가 성공적으로 저장되었습니다!")
    plt.show()

if __name__ == "__main__":
    train_and_evaluate()