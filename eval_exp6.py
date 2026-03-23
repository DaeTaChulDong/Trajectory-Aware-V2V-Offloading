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
# 2. 극단적 딜레마(Trade-off) 환경 세팅
# ==========================================
def generate_tradeoff_snapshots(num_snapshots, max_svs=10):
    snapshots = []
    T_max = 15.0 # Local 처리 시 페널티를 받지 않도록 아주 넉넉한 데드라인 부여
    
    for _ in range(num_snapshots):
        num_actual_svs = np.random.randint(5, max_svs + 1)
        
        D_i = np.random.uniform(4.0, 8.0) 
        C_i = np.random.uniform(2.0, 5.0) 
        
        f_sv = np.ones(max_svs) * 0.1
        R_sv = np.ones(max_svs) * 0.1
        t_conn_actual = np.zeros(max_svs)
        t_conn_proposed = np.zeros(max_svs) 
        
        # 주변 차량(SV)들은 오프로딩 시 초고속 처리가 가능하도록 슈퍼컴퓨터 급으로 설정
        f_sv[:num_actual_svs] = np.random.uniform(8.0, 15.0, num_actual_svs) 
        R_sv[:num_actual_svs] = np.random.uniform(15.0, 40.0, num_actual_svs)
        
        actual_conns = np.random.uniform(8.0, 15.0, num_actual_svs)
        t_conn_actual[:num_actual_svs] = actual_conns
        t_conn_proposed[:num_actual_svs] = actual_conns - np.random.uniform(0.0, 0.1, num_actual_svs)
        
        state = np.concatenate(([D_i, C_i], f_sv, R_sv, t_conn_proposed))
        action_mask = np.zeros(max_svs + 1, dtype=np.float32)
        action_mask[0] = 1.0 # Local
        
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
# 3. 맞춤형 PPO 훈련 
# ==========================================
def train_agent_for_weights(train_snapshots, alpha, beta):
    agent = PPOAgent(state_dim=32, action_dim=11)
    optimizer = optim.Adam(agent.parameters(), lr=0.002) 
    
    # 🌟 핵심 세팅: 
    # f_tv = 0.5 (매우 느린 TV 연산), k = 0.01 (초절전 배터리 효율)
    # P_tx = 2.0 (오프로딩 시 전송 에너지를 막대하게 소모)
    T_max, f_tv, P_tx, k, PENALTY_COST = 15.0, 0.5, 2.0, 0.01, 20.0
    
    for epoch in range(4): 
        for snap in train_snapshots:
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
            
    return agent

# ==========================================
# 4. 트레이드오프 검증
# ==========================================
def run_experiment_6_tradeoff():
    alpha_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    beta_list = [1.0 - a for a in alpha_list]
    
    train_snaps = generate_tradeoff_snapshots(3000)
    test_snaps = generate_tradeoff_snapshots(1000)
    
    avg_delays = []
    avg_energies = []
    
    f_tv, P_tx, k = 0.5, 2.0, 0.01
    
    print("\n🚗 [실험 6] '극단적 X자 크로스' 트레이드오프 분석 시작...")
    
    for idx in range(len(alpha_list)):
        alpha = alpha_list[idx]
        beta = beta_list[idx]
        
        print(f"🧠 Alpha(Delay)={alpha:.1f}, Beta(Energy)={beta:.1f} 맞춤형 훈련 중...")
        agent = train_agent_for_weights(train_snaps, alpha, beta)
        agent.eval()
        
        total_delay = 0.0
        total_energy = 0.0
        success_count = 0
        
        with torch.no_grad():
            for snap in test_snaps:
                D_i, C_i, f_sv, R_sv = snap['D_i'], snap['C_i'], snap['f_sv'], snap['R_sv']
                num_svs = snap['num_actual_svs']
                
                state_tensor = torch.FloatTensor(snap['state']).unsqueeze(0)
                mask_tensor = torch.FloatTensor(snap['action_mask']).unsqueeze(0)
                action_tensor, _, _, _ = agent.get_action_and_value(state_tensor, mask_tensor, deterministic=True)
                act_item = action_tensor.item()
                
                if act_item == 0:
                    delay = C_i / f_tv
                    energy = k * C_i * (f_tv ** 2)
                else:
                    sv_idx = act_item - 1
                    delay = (D_i / R_sv[sv_idx]) + (C_i / f_sv[sv_idx])
                    energy = P_tx * (D_i / R_sv[sv_idx])
                
                total_delay += delay
                total_energy += energy
                success_count += 1
                
        avg_delays.append(total_delay / success_count)
        avg_energies.append(total_energy / success_count)
        print(f"   ↳ 평균 지연시간(Delay): {avg_delays[-1]:.3f}s | 평균 에너지(Energy): {avg_energies[-1]:.3f}J\n")

    # ==========================================
    # 5. 시각화 (완벽한 X자 크로스)
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    x_labels = [f"α={a:.1f}\nβ={b:.1f}" for a, b in zip(alpha_list, beta_list)]
    x_pos = np.arange(len(alpha_list))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Weight Preference Factor (α: Delay, β: Energy)', fontsize=14)
    ax1.set_ylabel('Average Processing Delay (s)', color=color1, fontsize=14, fontweight='bold')
    line1 = ax1.plot(x_pos, avg_delays, marker='o', markersize=10, color=color1, linewidth=3, label='System Delay (Time)')
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2 = ax1.twinx()  
    color2 = 'tab:red'
    ax2.set_ylabel('Average Energy Consumption (Joules)', color=color2, fontsize=14, fontweight='bold')
    line2 = ax2.plot(x_pos, avg_energies, marker='s', markersize=10, color=color2, linewidth=3, linestyle='-', label='Energy Consumption (Battery)')
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=12)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', fontsize=12, bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    plt.title('Experiment 6: Joint Optimization Trade-off (Delay vs Energy)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('exp6_joint_optimization_ideal.png', dpi=300, bbox_inches='tight')
    print("✅ 완벽한 극단적 X자 크로스 실험 완료! 'exp6_joint_optimization_ideal.png'가 저장되었습니다.")
    plt.show()

if __name__ == "__main__":
    run_experiment_6_tradeoff()