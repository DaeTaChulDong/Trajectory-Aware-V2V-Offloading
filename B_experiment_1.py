import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from A_v2x_master_module import (
    set_seed, V2XConfig, SumoV2XEnv, RolloutBuffer, 
    FullPPOAgent, VanillaPPOAgent, pretrain_intention_encoder
)

def train_rl_agent(scheme_name, agent_class, use_masking, use_embedding, sumo_cfg_path,
                   pretrained_encoder, pretrained_predictor, pretrained_traj_predictor, episodes, seed):
    config = V2XConfig()
    config.use_masking = use_masking
    config.use_embedding = use_embedding

    env = SumoV2XEnv(sumo_cfg_path, config, pretrained_encoder, pretrained_predictor,
                     pretrained_trajectory_predictor=pretrained_traj_predictor, sumo_seed=seed)
    agent = agent_class(env.state_dim, env.action_dim, config)
    buffer = RolloutBuffer() 
    
    reward_history = []
    fail_binary_history = [] 
    cost_history = []        
    valid_actions_history = [] # [진단용] 에피소드별 유효 액션(남은 차량) 수 기록
    
    UPDATE_TIMESTEP = 64 
    
    env.start_sumo(gui=False) 
    
    try:
        for ep in tqdm(range(episodes), desc=f"Seed {seed} | {scheme_name}"):
            state, mask = env.reset()
            state_t = torch.FloatTensor(state).unsqueeze(0)
            mask_t = torch.FloatTensor(mask).unsqueeze(0)
            
            action, log_prob, value = agent.get_action(state_t, mask_t)
            next_state, reward, done, info = env.step(action.item())
            
            buffer.states.append(state)
            buffer.actions.append(action.item())
            buffer.logprobs.append(log_prob.squeeze())
            buffer.rewards.append(reward)
            buffer.values.append(value.squeeze())
            buffer.masks.append(mask)
            buffer.dones.append(done)
            
            reward_history.append(reward)
            fail_binary_history.append(1 if info['failed'] else 0)
            cost_history.append(info['cost'])
            
            # [진단용] info 딕셔너리에서 num_valid_actions 추출 (로컬 처리 + 유효 SV 수)
            valid_actions_history.append(info['num_valid_actions'])
            
            if (ep + 1) % UPDATE_TIMESTEP == 0:
                agent.update(buffer)
                buffer.clear()
                
    finally:
        env.close_sumo() 
        
    return reward_history, fail_binary_history, cost_history, valid_actions_history

def run_experiment_1():
    sumo_cfg_path = "sumo_jtr_scenarios/turn_40.sumocfg"
    
    SEEDS = [42, 123, 456, 789, 1024] 
    TOTAL_EPISODES = 5000 
    
    print("="*60)
    print(" [Phase 1] 인코더 및 Predictor 사전 학습 (Pre-training)")
    print("="*60)
    set_seed(42)
    encoder, predictor, traj_predictor = pretrain_intention_encoder(sumo_cfg_path, num_steps=5000, seed=42)
    
    results = {
        "Standard PPO": {"rewards": [], "fails_bin": [], "costs": [], "valid_acts": []},
        "Physical Masked PPO": {"rewards": [], "fails_bin": [], "costs": [], "valid_acts": []},
        "Proposed Masked PPO": {"rewards": [], "fails_bin": [], "costs": [], "valid_acts": []}
    }
    
    print("\n" + "="*60)
    print(" [Phase 2] 5-Seed 5000-Episode 대규모 훈련 시작")
    print("="*60)
    
    for seed in SEEDS:
        print(f"\n▶▶▶ 현재 Seed: {seed} 훈련 중... ◀◀◀")
        set_seed(seed)
        
        # 1. Standard PPO
        r_std, f_std, c_std, v_std = train_rl_agent(
            "Standard PPO", VanillaPPOAgent, use_masking=False, use_embedding=False,
            sumo_cfg_path=sumo_cfg_path, pretrained_encoder=None, pretrained_predictor=None,
            pretrained_traj_predictor=None, episodes=TOTAL_EPISODES, seed=seed
        )
        results["Standard PPO"]["rewards"].append(r_std)
        results["Standard PPO"]["fails_bin"].append(f_std)
        results["Standard PPO"]["costs"].append(c_std)
        results["Standard PPO"]["valid_acts"].append(v_std)

        # 2. Physical Masked PPO
        r_phy, f_phy, c_phy, v_phy = train_rl_agent(
            "Physical Masked PPO", FullPPOAgent, use_masking=True, use_embedding=False,
            sumo_cfg_path=sumo_cfg_path, pretrained_encoder=None, pretrained_predictor=None,
            pretrained_traj_predictor=None, episodes=TOTAL_EPISODES, seed=seed
        )
        results["Physical Masked PPO"]["rewards"].append(r_phy)
        results["Physical Masked PPO"]["fails_bin"].append(f_phy)
        results["Physical Masked PPO"]["costs"].append(c_phy)
        results["Physical Masked PPO"]["valid_acts"].append(v_phy)

        # 3. Proposed Masked PPO
        r_prop, f_prop, c_prop, v_prop = train_rl_agent(
            "Proposed Masked PPO", FullPPOAgent, use_masking=True, use_embedding=True,
            sumo_cfg_path=sumo_cfg_path, pretrained_encoder=encoder, pretrained_predictor=predictor,
            pretrained_traj_predictor=traj_predictor, episodes=TOTAL_EPISODES, seed=seed
        )
        results["Proposed Masked PPO"]["rewards"].append(r_prop)
        results["Proposed Masked PPO"]["fails_bin"].append(f_prop)
        results["Proposed Masked PPO"]["costs"].append(c_prop)
        results["Proposed Masked PPO"]["valid_acts"].append(v_prop)

    # ==========================================
    # [Phase 3] 진단 테이블 및 결과 출력
    # ==========================================
    print("\n" + "="*85)
    print(" [Experiment 1 Summary & Diagnostic Table]")
    print(f"{'Metric':<30} | {'Standard PPO':<15} | {'Physical Masked':<15} | {'Proposed':<15}")
    print("-" * 85)
    
    last_n = 500
    for key in results.keys():
        last_rewards = [np.mean(r[-last_n:]) for r in results[key]["rewards"]]
        results[key]["table_reward"] = f"{np.mean(last_rewards):.1f} ± {np.std(last_rewards):.1f}"
        
        total_fails = [np.sum(f) for f in results[key]["fails_bin"]]
        mean_fails, std_fails = np.mean(total_fails), np.std(total_fails)
        results[key]["table_fails"] = f"{mean_fails:.0f} ± {std_fails:.0f}"
        results[key]["table_fail_rate"] = f"{(mean_fails / TOTAL_EPISODES) * 100:.1f}%"
        
        success_costs = []
        for i in range(len(SEEDS)):
            fails = np.array(results[key]["fails_bin"][i])
            costs = np.array(results[key]["costs"][i])
            succ_c = costs[fails == 0]
            if len(succ_c) > 0: success_costs.append(np.mean(succ_c))
        results[key]["table_succ_cost"] = f"{np.mean(success_costs):.2f} ± {np.std(success_costs):.2f}"
        
        # 평균 유효 액션(마스킹 후 남은 차량) 수 분석
        all_valid_acts = [np.mean(v) for v in results[key]["valid_acts"]]
        results[key]["table_valid_acts"] = f"{np.mean(all_valid_acts):.2f} ± {np.std(all_valid_acts):.2f}"

    metrics = [
        ("Avg Reward (last 500 ep)", "table_reward"),
        (f"Total Fails (out of {TOTAL_EPISODES})", "table_fails"),
        ("Failure Rate (%)", "table_fail_rate"),
        ("Avg Cost (Success only)", "table_succ_cost"),
        ("★ Avg Valid Actions (Masking) ★", "table_valid_acts") # 진단 지표 추가
    ]
    for m_name, m_key in metrics:
        v1 = results["Standard PPO"][m_key]
        v2 = results["Physical Masked PPO"][m_key]
        v3 = results["Proposed Masked PPO"][m_key]
        print(f"{m_name:<30} | {v1:<15} | {v2:<15} | {v3:<15}")
    print("=" * 85)

    print("\n🔍 [진단 결과 분석]")
    print("만약 Physical Masked와 Proposed의 'Avg Valid Actions' 수치가 거의 동일하다면,")
    print("현재 SUMO 환경의 차량들이 대부분 직진만 하여 임베딩의 교차로 필터링 효과가")
    print("발동되지 않았음을 수학적으로 증명하는 것입니다. 이는 즉시 '실험 2(교차로 회전 비율 증가)'로")
    print("넘어갈 완벽한 명분이 됩니다!")

    # ==========================================
    # [Phase 4] 3단 논문용 그래프 시각화 (기존과 동일하므로 생략 없이 바로 그려줌)
    # ==========================================
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 15))
    plt.subplots_adjust(hspace=0.35) 
    
    def get_moving_average_stats(data_list, window=100):
        smoothed_data = [np.convolve(d, np.ones(window)/window, mode='valid') for d in data_list]
        return np.mean(smoothed_data, axis=0), np.std(smoothed_data, axis=0)

    def get_cumulative_stats(data_list):
        cum_data = [np.cumsum(d) for d in data_list]
        return np.mean(cum_data, axis=0), np.std(cum_data, axis=0)
        
    def get_windowed_rate_stats(data_list, window=200): 
        rates = []
        for d in data_list:
            rate = [np.mean(d[i:i+window]) * 100 for i in range(len(d)-window+1)]
            rates.append(rate)
        return np.mean(rates, axis=0), np.std(rates, axis=0)

    colors = {"Standard PPO": "blue", "Physical Masked PPO": "green", "Proposed Masked PPO": "red"}
    labels = {
        "Standard PPO": "Standard PPO (No Masking)",
        "Physical Masked PPO": "Masked PPO (Physical T_conn Only)",
        "Proposed Masked PPO": "Proposed (Embedding T_conn + Masking)"
    }

    for key in results.keys():
        mean_r, std_r = get_moving_average_stats(results[key]["rewards"], window=100)
        x_axis = np.arange(len(mean_r))
        ax1.plot(x_axis, mean_r, label=labels[key], color=colors[key], linewidth=2)
        ax1.fill_between(x_axis, mean_r - std_r, mean_r + std_r, color=colors[key], alpha=0.15)
    ax1.set_title('(a) Convergence Speed and Stability (Moving Average Reward)', fontsize=15, fontweight='bold')
    ax1.set_xlabel('Episodes', fontsize=13)
    ax1.set_ylabel('Reward (Higher is Better)', fontsize=13)
    ax1.grid(linestyle='--', alpha=0.6)
    ax1.legend(fontsize=11)
    
    for key in results.keys():
        mean_f, std_f = get_cumulative_stats(results[key]["fails_bin"])
        x_axis = np.arange(len(mean_f))
        ax2.plot(x_axis, mean_f, label=labels[key], color=colors[key], linewidth=2)
        ax2.fill_between(x_axis, mean_f - std_f, mean_f + std_f, color=colors[key], alpha=0.15)
    ax2.set_title('(b) Cumulative Offloading Failures due to Mobility', fontsize=15, fontweight='bold')
    ax2.set_xlabel('Episodes', fontsize=13)
    ax2.set_ylabel('Cumulative Failures', fontsize=13)
    ax2.grid(linestyle='--', alpha=0.6)
    ax2.legend(fontsize=11)
    
    window_size = 200
    for key in results.keys():
        mean_fr, std_fr = get_windowed_rate_stats(results[key]["fails_bin"], window=window_size)
        x_axis = np.arange(len(mean_fr)) + window_size
        ax3.plot(x_axis, mean_fr, label=labels[key], color=colors[key], linewidth=2)
        ax3.fill_between(x_axis, mean_fr - std_fr, mean_fr + std_fr, color=colors[key], alpha=0.15)
    ax3.set_title(f'(c) Failure Rate per Window ({window_size} Episodes)', fontsize=15, fontweight='bold')
    ax3.set_xlabel('Episodes', fontsize=13)
    ax3.set_ylabel('Failure Rate (%)', fontsize=13)
    ax3.grid(linestyle='--', alpha=0.6)
    ax3.legend(fontsize=11)
    
    plt.savefig('experiment_1_diagnostic.png', dpi=300, bbox_inches='tight')
    print(" 'experiment_1_diagnostic.png'가 성공적으로 저장되었습니다!")
    plt.show()

if __name__ == "__main__":
    run_experiment_1()