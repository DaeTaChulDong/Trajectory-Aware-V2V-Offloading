import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# 1. PPO 에이전트
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
# 2. 현실적인 SUMO 데이터셋 (Hard Mode)
# ==========================================
def generate_realistic_snapshots(num_snapshots, target_turn_ratio, max_svs=10):
    snapshots = []
    T_max = 5.0 
    
    for _ in range(num_snapshots):
        num_actual_svs = np.random.randint(5, max_svs + 1)
        
        # 태스크 크기 대폭 증가 (연산 시간이 오래 걸리게 유도)
        D_i = np.random.uniform(5.0, 15.0) 
        C_i = np.random.uniform(2.0, 6.0) 
        
        f_sv = np.ones(max_svs)
        R_sv = np.ones(max_svs)
        t_conn_actual = np.zeros(max_svs)
        t_conn_proposed = np.zeros(max_svs) 
        
        distances = np.random.uniform(10.0, 150.0, num_actual_svs)
        is_turning = np.random.rand(num_actual_svs) < target_turn_ratio 
        
        f_sv[:num_actual_svs] = np.random.uniform(2.0, 6.0, num_actual_svs) 
        
        base_rates = 60.0 / (distances / 10.0) 
        actual_rates = np.where(is_turning, base_rates * np.random.uniform(0.1, 0.4), base_rates * np.random.uniform(0.8, 1.2))
        R_sv[:num_actual_svs] = np.clip(actual_rates, 1.0, 40.0)
        
        # 꺾이는 차량의 연결 시간 대폭 단축 (지뢰밭 강화)
        actual_conns = np.where(is_turning, np.random.uniform(0.2, 1.2, num_actual_svs), np.random.uniform(3.0, 8.0, num_actual_svs))
        t_conn_actual[:num_actual_svs] = actual_conns
        t_conn_proposed[:num_actual_svs] = actual_conns - np.random.uniform(0.0, 0.2, num_actual_svs)
        
        state = np.concatenate(([D_i, C_i], f_sv, R_sv, t_conn_proposed))
        action_mask = np.zeros(max_svs + 1, dtype=np.float32)
        action_mask[0] = 1.0 
        
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
# 3. PPO 사전 학습 (Base Training)
# ==========================================
def train_ppo_for_evaluation():
    print("🧠 [Phase 1] 평가를 위한 PPO 에이전트 사전 학습 진행 중 (Hard Mode)...")
    base_snapshots = generate_realistic_snapshots(4000, target_turn_ratio=0.3)
    agent = PPOAgent(state_dim=32, action_dim=11)
    optimizer = optim.Adam(agent.parameters(), lr=0.001) 
    T_max, f_tv, P_tx, k, alpha, beta, PENALTY_COST = 5.0, 1.5, 0.5, 0.02, 0.5, 0.5, 10.0
    
    for epoch in range(3): 
        for snap in base_snapshots:
            state = torch.FloatTensor(snap['state']).unsqueeze(0)
            action_mask = torch.FloatTensor(snap['action_mask']).unsqueeze(0)
            action, log_prob, _, value = agent.get_action_and_value(state, action_mask)
            act_item = action.item()
            
            D_i, C_i, f_sv, R_sv, t_conn_actual = snap['D_i'], snap['C_i'], snap['f_sv'], snap['R_sv'], snap['t_conn_actual']
            is_failed, final_cost = False, 0.0
            
            if act_item == 0: 
                T_local = C_i / f_tv
                if T_local > T_max: is_failed = True
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
            
    print("✅ PPO 에이전트 학습 완료!")
    return agent

# ==========================================
# 4. 강건성 실제 평가 및 누적 데이터 수집
# ==========================================
def evaluate_robustness(trained_agent):
    turn_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    num_trials = 2000 
    max_svs = 10 
    
    T_max, f_tv, P_tx, k, alpha, beta = 5.0, 1.5, 0.5, 0.02, 0.5, 0.5
    
    rates_prop, rates_lin, rates_gre = [], [], []
    
    # 60% 가혹 환경에서의 누적 실패(Drop) 횟수 추적 변수
    cum_drops_prop = []
    cum_drops_lin = []
    cum_drops_gre = []
    
    print("\n🚗 [Phase 2] 통제된 회전 비율 환경에서 실제 신경망 추론 테스트 시작...")
    trained_agent.eval() 
    
    with torch.no_grad():
        for tr in turn_ratios:
            test_snapshots = generate_realistic_snapshots(num_trials, target_turn_ratio=tr)
            success_prop, success_lin, success_gre = 0, 0, 0
            
            drop_count_prop, drop_count_lin, drop_count_gre = 0, 0, 0
            
            for snap in test_snapshots:
                D_i, C_i, f_sv, R_sv = snap['D_i'], snap['C_i'], snap['f_sv'], snap['R_sv']
                t_conn_actual = snap['t_conn_actual']
                num_svs = snap['num_actual_svs']
                
                T_total_svs = np.full(max_svs, np.inf) 
                T_total_svs[:num_svs] = (D_i / R_sv[:num_svs]) + (C_i / f_sv[:num_svs])
                T_local = C_i / f_tv
                
                # 1. Proposed
                state_tensor = torch.FloatTensor(snap['state']).unsqueeze(0)
                mask_tensor = torch.FloatTensor(snap['action_mask']).unsqueeze(0)
                action_tensor, _, _, _ = trained_agent.get_action_and_value(state_tensor, mask_tensor, deterministic=True)
                act_item = action_tensor.item()
                
                is_prop_success = False
                if act_item == 0:
                    if T_local <= T_max: is_prop_success = True
                else:
                    sv_idx = act_item - 1
                    if sv_idx < num_svs and T_total_svs[sv_idx] <= T_max and T_total_svs[sv_idx] <= t_conn_actual[sv_idx]:
                        is_prop_success = True
                
                if is_prop_success: success_prop += 1
                else: drop_count_prop += 1

                # 2. Greedy-Comm
                is_gre_success = False
                idx_greedy = np.argmax(R_sv[:num_svs]) 
                if T_total_svs[idx_greedy] <= T_max and T_total_svs[idx_greedy] <= t_conn_actual[idx_greedy]:
                    is_gre_success = True
                
                if is_gre_success: success_gre += 1
                else: drop_count_gre += 1
                    
                # 3. Linear-Predict
                is_lin_success = False
                t_conn_linear = np.random.uniform(4.0, 8.0, max_svs) 
                mask_lin = (T_total_svs[:num_svs] <= T_max) & (T_total_svs[:num_svs] <= t_conn_linear[:num_svs])
                if np.any(mask_lin):
                    Cost_svs = alpha * T_total_svs[:num_svs] + beta * (P_tx * (D_i / R_sv[:num_svs]))
                    valid_costs = np.where(mask_lin, Cost_svs, np.inf)
                    idx_lin = np.argmin(valid_costs)
                    if T_total_svs[idx_lin] <= t_conn_actual[idx_lin]:
                        is_lin_success = True
                else:
                    if T_local <= T_max: is_lin_success = True
                
                if is_lin_success: success_lin += 1
                else: drop_count_lin += 1

                # 60% 환경일 때 매 시도마다 누적 드롭 횟수 저장
                if abs(tr - 0.6) < 0.01:
                    cum_drops_prop.append(drop_count_prop)
                    cum_drops_lin.append(drop_count_lin)
                    cum_drops_gre.append(drop_count_gre)

            rates_prop.append((success_prop / num_trials) * 100)
            rates_lin.append((success_lin / num_trials) * 100)
            rates_gre.append((success_gre / num_trials) * 100)
            
            print(f"Turn Ratio {tr*100:2.0f}% | Neural PPO: {rates_prop[-1]:.1f}% | Linear: {rates_lin[-1]:.1f}% | Greedy: {rates_gre[-1]:.1f}%")

    # ==========================================
    # 5. 듀얼 플롯 시각화 (논문 방어용 궁극기)
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    x_axis = [tr * 100 for tr in turn_ratios]
    
    # [왼쪽] 기존 성공률 그래프
    ax1.plot(x_axis, rates_prop, marker='o', markersize=10, linewidth=3, color='red', label='Proposed (Neural PPO Inference)')
    ax1.plot(x_axis, rates_lin, marker='s', markersize=9, linewidth=2.5, color='green', linestyle='--', label='Linear-Predict (Assumption: Straight)')
    ax1.plot(x_axis, rates_gre, marker='^', markersize=9, linewidth=2.5, color='blue', linestyle='-.', label='Greedy-Comm (Highest SNR)')
    ax1.set_title('Robustness against Intersection Uncertainties', fontsize=14)
    ax1.set_xlabel('Vehicle Turn Ratio at SUMO Intersection (%)', fontsize=12)
    ax1.set_ylabel('Task Completion Ratio (%)', fontsize=12)
    ax1.set_ylim(0, 105)
    ax1.legend(loc='lower left')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # [오른쪽] 🌟 극악 환경(60%)에서의 누적 태스크 드롭 횟수 (오타 수정 완)
    trials_axis = np.arange(num_trials)
    ax2.plot(trials_axis, cum_drops_gre, label='Greedy-Comm', color='blue', linestyle='-.', linewidth=2.5) 
    ax2.plot(trials_axis, cum_drops_lin, label='Linear-Predict', color='green', linestyle='--', linewidth=2.5)
    ax2.plot(trials_axis, cum_drops_prop, label='Proposed Masked PPO', color='red', linewidth=3)
    ax2.set_title('Cumulative Task Failures (Drops) at 60% Turn Ratio', fontsize=14)
    ax2.set_xlabel('Number of Evaluation Trials', fontsize=12)
    ax2.set_ylabel('Cumulative Dropped Tasks', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.suptitle('Experiment 2: Inference Robustness and Cumulative Failures', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('exp2_hardmode_dual_plot.png', dpi=300)
    print("\n✅ 에러 수정 및 누적 플롯 적용 완료! 'exp2_hardmode_dual_plot.png'가 저장되었습니다.")
    plt.show()

if __name__ == "__main__":
    trained_agent = train_ppo_for_evaluation()
    evaluate_robustness(trained_agent)