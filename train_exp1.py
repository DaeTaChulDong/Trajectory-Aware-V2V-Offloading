import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# 1. PPO 에이전트 및 버퍼 (이전과 완전히 동일)
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

    def get_action_and_value(self, state, action_mask=None, action=None):
        logits = self.actor(state)
        if action_mask is not None:
            logits = logits - (1.0 - action_mask) * 1e9 
            
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(state)

class PPOBuffer:
    def __init__(self):
        self.states, self.actions, self.logprobs, self.rewards, self.masks = [], [], [], [], []
    def clear(self):
        self.states.clear(); self.actions.clear(); self.logprobs.clear(); self.rewards.clear(); self.masks.clear()

# ==========================================
# 2. 데이터셋 생성 (이전과 동일)
# ==========================================
def generate_mock_snapshots(num_snapshots=5000, max_svs=10):
    snapshots = []
    T_max = 5.0 
    for _ in range(num_snapshots):
        num_actual_svs = np.random.randint(4, max_svs + 1)
        D_i = np.random.uniform(2.0, 8.0) 
        C_i = np.random.uniform(1.0, 4.0) 
        f_sv = np.zeros(max_svs)
        R_sv = np.zeros(max_svs)
        t_conn_actual = np.zeros(max_svs)
        t_conn_proposed = np.zeros(max_svs) 
        is_turning = np.random.rand(num_actual_svs) < 0.3 
        f_sv[:num_actual_svs] = np.random.uniform(1.0, 8.0, num_actual_svs) 
        R_sv[:num_actual_svs] = np.random.uniform(5.0, 50.0, num_actual_svs)
        actual_conns = np.where(is_turning, np.random.uniform(0.5, 2.0, num_actual_svs), np.random.uniform(4.0, 8.0, num_actual_svs))
        t_conn_actual[:num_actual_svs] = actual_conns
        t_conn_proposed[:num_actual_svs] = actual_conns - np.random.uniform(0.0, 0.2, num_actual_svs)
        state = np.concatenate(([D_i, C_i], f_sv, R_sv, t_conn_proposed))
        action_mask = np.zeros(max_svs + 1, dtype=np.float32)
        action_mask[0] = 1.0 
        for j in range(num_actual_svs):
            t_total = (D_i / R_sv[j]) + (C_i / f_sv[j])
            if (t_total <= T_max) and (t_total <= t_conn_proposed[j]):
                action_mask[j+1] = 1.0
        snapshots.append({'state': state, 'action_mask': action_mask, 'D_i': D_i, 'C_i': C_i, 'f_sv': f_sv, 'R_sv': R_sv, 't_conn_actual': t_conn_actual, 'num_actual_svs': num_actual_svs})
    return snapshots

# ==========================================
# 3. PPO 학습 알고리즘 (이전과 동일)
# ==========================================
def train_ppo(snapshots, use_masking):
    agent = PPOAgent(state_dim=32, action_dim=11)
    optimizer = optim.Adam(agent.parameters(), lr=0.001) 
    buffer = PPOBuffer()
    T_max, f_tv, P_tx, k, alpha, beta, PENALTY_COST = 5.0, 1.0, 0.5, 0.02, 0.5, 0.5, 8.0
    batch_size, update_epochs, clip_ratio, entropy_coef = 64, 4, 0.2, 0.005
    costs_history = []
    
    for step_idx, snap in enumerate(snapshots):
        state = torch.FloatTensor(snap['state']).unsqueeze(0)
        action_mask = torch.FloatTensor(snap['action_mask']).unsqueeze(0) if use_masking else None
        action, log_prob, _, value = agent.get_action_and_value(state, action_mask)
        act_item = action.item()
        
        D_i, C_i, f_sv, R_sv, t_conn_actual = snap['D_i'], snap['C_i'], snap['f_sv'], snap['R_sv'], snap['t_conn_actual']
        is_failed, final_cost = False, 0.0
        
        if act_item == 0: 
            T_local = C_i / f_tv
            final_cost = alpha * T_local + beta * (k * C_i * (f_tv ** 2))
            if T_local > T_max: is_failed = True
        else: 
            sv_idx = act_item - 1
            if sv_idx >= snap['num_actual_svs']: 
                is_failed = True 
            else:
                T_total = (D_i / R_sv[sv_idx]) + (C_i / f_sv[sv_idx])
                final_cost = alpha * T_total + beta * (P_tx * (D_i / R_sv[sv_idx]))
                if T_total > T_max or T_total > t_conn_actual[sv_idx]: is_failed = True
        
        record_cost = PENALTY_COST if is_failed else final_cost
        costs_history.append(record_cost)
        
        buffer.states.append(snap['state']); buffer.actions.append(act_item); buffer.logprobs.append(log_prob.squeeze().detach())
        buffer.rewards.append(-record_cost); buffer.masks.append(snap['action_mask'] if use_masking else None)
        
        if (step_idx + 1) % batch_size == 0:
            old_states = torch.FloatTensor(np.array(buffer.states))
            old_actions = torch.LongTensor(buffer.actions)
            old_logprobs = torch.stack(buffer.logprobs)
            rewards = torch.FloatTensor(buffer.rewards)
            masks = torch.FloatTensor(np.array(buffer.masks)) if use_masking else None
            
            with torch.no_grad():
                _, _, _, old_values = agent.get_action_and_value(old_states, masks, old_actions)
                advantages = (rewards - old_values.squeeze())
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            for _ in range(update_epochs):
                _, new_logprobs, entropy, new_values = agent.get_action_and_value(old_states, masks, old_actions)
                ratios = torch.exp(new_logprobs.squeeze() - old_logprobs)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(new_values.squeeze(), rewards)
                loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy.mean() 
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            buffer.clear()
            
    return costs_history

# ==========================================
# 4. 🌟 다중 시드(Multi-Seed) 평가 및 신뢰 구간 시각화
# ==========================================
def run_multiple_seeds(snapshots, use_masking, num_seeds=5):
    all_costs = []
    print(f"\n▶ {'Masked PPO' if use_masking else 'Standard PPO'} {num_seeds}회 반복 학습 진행 중...")
    for seed in tqdm(range(num_seeds)):
        # 파이토치 및 넘파이 시드 고정을 통해 에이전트 초기화 다양성 부여 (스냅샷은 고정)
        torch.manual_seed(seed * 10)
        costs = train_ppo(snapshots, use_masking)
        all_costs.append(costs)
    return np.array(all_costs) # Shape: (num_seeds, num_episodes)

if __name__ == "__main__":
    snapshots = generate_mock_snapshots(num_snapshots=5000)
    
    # 각 알고리즘을 3번씩 반복 학습
    num_runs = 3
    costs_prop_all = run_multiple_seeds(snapshots, use_masking=True, num_seeds=num_runs)
    costs_base_all = run_multiple_seeds(snapshots, use_masking=False, num_seeds=num_runs)
    
    def moving_average_2d(data_2d, window_size=200):
        # 여러 시드의 결과를 각각 이동평균 처리
        smoothed = [np.convolve(row, np.ones(window_size)/window_size, mode='valid') for row in data_2d]
        return np.array(smoothed)
        
    smooth_prop = moving_average_2d(costs_prop_all)
    smooth_base = moving_average_2d(costs_base_all)
    
    # 평균(Mean)과 표준편차(Std) 계산
    prop_mean, prop_std = np.mean(smooth_prop, axis=0), np.std(smooth_prop, axis=0)
    base_mean, base_std = np.mean(smooth_base, axis=0), np.std(smooth_base, axis=0)
    
    # 누적 비용도 다중 시드 평균 계산
    cum_prop_mean = np.mean([np.cumsum(row) for row in costs_prop_all], axis=0)
    cum_base_mean = np.mean([np.cumsum(row) for row in costs_base_all], axis=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    x_axis = np.arange(len(prop_mean))
    
    # [왼쪽] 학습 곡선 (신뢰 구간 포함)
    ax1.plot(x_axis, base_mean, label='Standard PPO', color='blue', alpha=0.8)
    ax1.fill_between(x_axis, base_mean - base_std, base_mean + base_std, color='blue', alpha=0.2)
    
    ax1.plot(x_axis, prop_mean, label='Proposed Masked PPO', color='red', linewidth=2.5)
    ax1.fill_between(x_axis, prop_mean - prop_std, prop_mean + prop_std, color='red', alpha=0.2)
    
    ax1.set_title('Average Cost per Episode (with 1-Std Confidence Interval)', fontsize=14)
    ax1.set_xlabel('Episodes', fontsize=12)
    ax1.set_ylabel('Moving Average Cost', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # [오른쪽] 누적 비용 평균
    x_axis_cum = np.arange(len(cum_prop_mean))
    ax2.plot(x_axis_cum, cum_base_mean, label='Standard PPO', color='blue', linewidth=2)
    ax2.plot(x_axis_cum, cum_prop_mean, label='Proposed Masked PPO', color='red', linewidth=2.5)
    ax2.set_title('Cumulative Cost over Training (Averaged)', fontsize=14)
    ax2.set_xlabel('Episodes', fontsize=12)
    ax2.set_ylabel('Cumulative Total Cost', fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.suptitle('Experiment 1: Robust Convergence Speed and Efficiency (Multi-Seed)', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('exp1_final.png', dpi=300, bbox_inches='tight')
    print("\n 'exp1_final.png'가 저장되었습니다.")
    plt.show()