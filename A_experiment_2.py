import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from A_v2x_master_module import (
    set_seed, V2XConfig, SumoV2XEnv, RolloutBuffer, 
    FullPPOAgent, GreedySNRAgent, GreedyStabilityAgent, pretrain_intention_encoder
)

def run_agent_episodes(scheme_name, agent_class, use_masking, use_embedding, sumo_cfg_path, 
                       pretrained_encoder, pretrained_predictor, episodes, seed, is_rl=True):
    config = V2XConfig()
    config.use_masking = use_masking 
    config.use_embedding = use_embedding 
    
    env = SumoV2XEnv(sumo_cfg_path, config, pretrained_encoder, pretrained_predictor, sumo_seed=seed)
    agent = agent_class(env.state_dim, env.action_dim, config)
    buffer = RolloutBuffer() if is_rl else None
    
    fail_binary_history = [] 
    cost_history = []        
    UPDATE_TIMESTEP = 64 
    
    env.start_sumo(gui=False) 
    try:
        for ep in tqdm(range(episodes), desc=f"Seed {seed} | {scheme_name}"):
            state, mask = env.reset()
            state_t = torch.FloatTensor(state).unsqueeze(0)
            mask_t = torch.FloatTensor(mask).unsqueeze(0)
            
            if is_rl:
                action, log_prob, value = agent.get_action(state_t, mask_t)
            else:
                action, _, _ = agent.get_action(state_t, mask_t, deterministic=True)
                
            next_state, reward, done, info = env.step(action.item())
            
            fail_binary_history.append(1 if info['failed'] else 0)
            cost_history.append(info['cost'])
            
            if is_rl:
                buffer.states.append(state)
                buffer.actions.append(action.item())
                buffer.logprobs.append(log_prob.squeeze())
                buffer.rewards.append(reward)
                buffer.values.append(value.squeeze())
                buffer.masks.append(mask)
                buffer.dones.append(done)
                
                if (ep + 1) % UPDATE_TIMESTEP == 0:
                    agent.update(buffer)
                    buffer.clear()
    finally:
        env.close_sumo() 
        
    eval_window = 500
    eval_fails = fail_binary_history[-eval_window:]
    eval_costs = cost_history[-eval_window:]
    
    return eval_fails, eval_costs

def run_experiment_2():
    TURN_RATIOS = [10, 25, 40, 55, 70]
    SEEDS = [42, 123, 456] 
    EPISODES = 2000 
    
    results = {
        "Proposed Masked PPO": {"success_rates": [], "avg_costs": []},
        "Physical Masked PPO": {"success_rates": [], "avg_costs": []},
        "Greedy-SNR": {"success_rates": [], "avg_costs": []},
        "Greedy-Stability": {"success_rates": [], "avg_costs": []}
    }
    
    print("="*70)
    print("🌟 [Experiment 2] 교차로 회전 비율(Turn Ratio) 변화에 따른 성능 평가 (Fair Baseline 적용)")
    print("="*70)
    
    for tr in TURN_RATIOS:
        cfg_path = f"sumo_jtr_scenarios/turn_{tr:02d}.sumocfg"
        print(f"\n\n{'='*50}")
        print(f"🚀 [Turn Ratio: {tr}%] 시나리오 분석 시작")
        print(f"{'='*50}")
        
        set_seed(42)
        encoder, predictor = pretrain_intention_encoder(cfg_path, num_steps=5000, seed=42)
        
        temp_results = {scheme: {"sr": [], "cost": []} for scheme in results.keys()}
        
        for seed in SEEDS:
            print(f"\n▶ Seed {seed} 실행 중...")
            set_seed(seed)
            
            # A. Proposed (임베딩 O, 마스킹 O)
            f_prop, c_prop = run_agent_episodes(
                "Proposed", FullPPOAgent, True, True, cfg_path, encoder, predictor, EPISODES, seed, is_rl=True
            )
            
            # B. Physical (임베딩 X, 마스킹 O)
            f_phy, c_phy = run_agent_episodes(
                "Physical", FullPPOAgent, True, False, cfg_path, None, None, EPISODES, seed, is_rl=True
            )
            
            # 🌟 [수정 핵심] C. Greedy-SNR (임베딩 X, 마스킹 O) -> 공정한 비교!
            f_snr, c_snr = run_agent_episodes(
                "Greedy-SNR", GreedySNRAgent, True, False, cfg_path, None, None, 1000, seed, is_rl=False
            )
            
            # 🌟 [수정 핵심] D. Greedy-Stability (임베딩 X, 마스킹 O) -> 공정한 비교!
            f_stab, c_stab = run_agent_episodes(
                "Greedy-Stability", GreedyStabilityAgent, True, False, cfg_path, None, None, 1000, seed, is_rl=False
            )
            
            def calc_metrics(fails, costs):
                fails_np, costs_np = np.array(fails), np.array(costs)
                success_rate = (1.0 - np.mean(fails_np)) * 100
                succ_costs = costs_np[fails_np == 0]
                avg_cost = np.mean(succ_costs) if len(succ_costs) > 0 else 0
                return success_rate, avg_cost

            sr, cost = calc_metrics(f_prop, c_prop)
            temp_results["Proposed Masked PPO"]["sr"].append(sr); temp_results["Proposed Masked PPO"]["cost"].append(cost)
            
            sr, cost = calc_metrics(f_phy, c_phy)
            temp_results["Physical Masked PPO"]["sr"].append(sr); temp_results["Physical Masked PPO"]["cost"].append(cost)
            
            sr, cost = calc_metrics(f_snr, c_snr)
            temp_results["Greedy-SNR"]["sr"].append(sr); temp_results["Greedy-SNR"]["cost"].append(cost)
            
            sr, cost = calc_metrics(f_stab, c_stab)
            temp_results["Greedy-Stability"]["sr"].append(sr); temp_results["Greedy-Stability"]["cost"].append(cost)
            
        for scheme in results.keys():
            results[scheme]["success_rates"].append(np.mean(temp_results[scheme]["sr"]))
            results[scheme]["avg_costs"].append(np.mean(temp_results[scheme]["cost"]))

    print("\n📊 모든 훈련 완료! 논문용 그래프를 생성합니다...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = {
        "Proposed Masked PPO": "red", "Physical Masked PPO": "green", 
        "Greedy-SNR": "blue", "Greedy-Stability": "purple"
    }
    markers = {"Proposed Masked PPO": "o", "Physical Masked PPO": "s", "Greedy-SNR": "^", "Greedy-Stability": "D"}
    
    x_labels = [f"{tr}%" for tr in TURN_RATIOS]
    
    for scheme, data in results.items():
        ax1.plot(x_labels, data["success_rates"], label=scheme, color=colors[scheme], marker=markers[scheme], linewidth=2, markersize=8)
    
    ax1.set_title('(a) Offloading Success Rate vs. Turn Ratio', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Intersection Turn Ratio', fontsize=12)
    ax1.set_ylabel('Success Rate (%)', fontsize=12)
    ax1.grid(linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 105)
    
    for scheme, data in results.items():
        valid_costs = [c if c > 0 else np.nan for c in data["avg_costs"]]
        ax2.plot(x_labels, valid_costs, label=scheme, color=colors[scheme], marker=markers[scheme], linewidth=2, markersize=8)
        
    ax2.set_title('(b) Average Cost vs. Turn Ratio (Successful Tasks Only)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Intersection Turn Ratio', fontsize=12)
    ax2.set_ylabel('Average Cost', fontsize=12)
    ax2.grid(linestyle='--', alpha=0.7)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('experiment_2_fair_baseline.png', dpi=300)
    print("✅ 'experiment_2_fair_baseline.png' 저장 완료!")
    plt.show()

if __name__ == "__main__":
    run_experiment_2()