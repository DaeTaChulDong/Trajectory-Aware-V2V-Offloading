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
                       pretrained_encoder, pretrained_predictor, pretrained_traj_predictor,
                       episodes, seed, is_rl=True, model_save_path=None, model_load_path=None):
    config = V2XConfig()
    config.use_masking = use_masking
    config.use_embedding = use_embedding

    env = SumoV2XEnv(sumo_cfg_path, config, pretrained_encoder, pretrained_predictor,
                     pretrained_trajectory_predictor=pretrained_traj_predictor, sumo_seed=seed)
    agent = agent_class(env.state_dim, env.action_dim, config)

    eval_window = 500
    fail_binary_history = []
    cost_history = []
    fail_reason_history = []

    # 저장된 모델이 있으면 로드 후 평가만 수행
    if model_load_path is not None and os.path.exists(model_load_path):
        agent.load_state_dict(torch.load(model_load_path))
        print(f"  [로드] {scheme_name} 모델 로드: {model_load_path}")
        env.start_sumo(gui=False)
        try:
            for ep in tqdm(range(eval_window), desc=f"Seed {seed} | {scheme_name} [Eval]"):
                state, mask = env.reset()
                state_t = torch.FloatTensor(state).unsqueeze(0)
                mask_t = torch.FloatTensor(mask).unsqueeze(0)
                action, _, _ = agent.get_action(state_t, mask_t, deterministic=True)
                _, _, _, info = env.step(action.item())
                fail_binary_history.append(1 if info['failed'] else 0)
                cost_history.append(info['cost'])
                fail_reason_history.append(info.get('fail_reason', 'none'))
        finally:
            env.close_sumo()
        return fail_binary_history, cost_history, fail_reason_history

    buffer = RolloutBuffer() if is_rl else None
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
            fail_reason_history.append(info.get('fail_reason', 'none'))

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

    if is_rl and model_save_path is not None:
        torch.save(agent.state_dict(), model_save_path)
        print(f"  [저장] {scheme_name} 모델 저장: {model_save_path}")

    eval_fails = fail_binary_history[-eval_window:]
    eval_costs = cost_history[-eval_window:]
    eval_fail_reasons = fail_reason_history[-eval_window:]

    return eval_fails, eval_costs, eval_fail_reasons


def run_experiment_2():
    TURN_RATIOS = [10, 25, 40, 55, 70]
    SEEDS = [42, 123, 456]
    EPISODES = 2000

    os.makedirs('saved_models', exist_ok=True)

    results = {
        "Proposed (Emb+Mask)":          {"success_rates": [], "avg_costs": [], "all_fail_reasons": []},
        "Physical (Mask only)":          {"success_rates": [], "avg_costs": [], "all_fail_reasons": []},
        "Greedy-Comm (SNR)":             {"success_rates": [], "avg_costs": [], "all_fail_reasons": []},
        "Linear Prediction (Stability)": {"success_rates": [], "avg_costs": [], "all_fail_reasons": []},
    }

    print("="*70)
    print("🌟 [Experiment 2] 교차로 회전 비율(Turn Ratio) 변화에 따른 성능 평가")
    print("="*70)

    print("\n 인코더 사전학습 (turn_40 기준, 모든 시나리오에 공통 적용)")
    set_seed(42)
    encoder, predictor, traj_predictor = pretrain_intention_encoder(
        "sumo_jtr_scenarios/turn_40.sumocfg", num_steps=8000, seed=42)

    for tr in TURN_RATIOS:
        cfg_path = f"sumo_jtr_scenarios/turn_{tr:02d}.sumocfg"
        print(f"\n\n{'='*50}")
        print(f" [Turn Ratio: {tr}%] 시나리오 분석 시작")
        print(f"{'='*50}")

        temp_results = {scheme: {"sr": [], "cost": [], "fail_reasons": []} for scheme in results.keys()}

        for seed in SEEDS:
            print(f"\n▶ Seed {seed} 실행 중...")
            set_seed(seed)

            # A. Proposed (임베딩 O, 마스킹 O)
            prop_path = f'saved_models/exp2_Proposed_turn{tr}_seed{seed}.pt'
            f_prop, c_prop, r_prop = run_agent_episodes(
                "Proposed", FullPPOAgent, True, True, cfg_path,
                encoder, predictor, traj_predictor,
                EPISODES, seed, is_rl=True,
                model_save_path=prop_path,
                model_load_path=prop_path if os.path.exists(prop_path) else None,
            )

            # B. Physical (임베딩 X, 마스킹 O)
            phy_path = f'saved_models/exp2_Physical_turn{tr}_seed{seed}.pt'
            f_phy, c_phy, r_phy = run_agent_episodes(
                "Physical", FullPPOAgent, True, False, cfg_path,
                None, None, None,
                EPISODES, seed, is_rl=True,
                model_save_path=phy_path,
                model_load_path=phy_path if os.path.exists(phy_path) else None,
            )

            # C. Greedy-SNR (저장/로드 불필요)
            f_snr, c_snr, r_snr = run_agent_episodes(
                "Greedy-SNR", GreedySNRAgent, True, False, cfg_path,
                None, None, None,
                EPISODES, seed, is_rl=False,
            )

            # D. Greedy-Stability (저장/로드 불필요)
            f_stab, c_stab, r_stab = run_agent_episodes(
                "Greedy-Stability", GreedyStabilityAgent, True, False, cfg_path,
                None, None, None,
                EPISODES, seed, is_rl=False,
            )

            def calc_metrics(fails, costs):
                fails_np, costs_np = np.array(fails), np.array(costs)
                success_rate = (1.0 - np.mean(fails_np)) * 100
                succ_costs = costs_np[fails_np == 0]
                avg_cost = np.mean(succ_costs) if len(succ_costs) > 0 else 0.0
                return success_rate, avg_cost

            sr, cost = calc_metrics(f_prop, c_prop)
            temp_results["Proposed (Emb+Mask)"]["sr"].append(sr)
            temp_results["Proposed (Emb+Mask)"]["cost"].append(cost)
            temp_results["Proposed (Emb+Mask)"]["fail_reasons"].extend(r_prop)

            sr, cost = calc_metrics(f_phy, c_phy)
            temp_results["Physical (Mask only)"]["sr"].append(sr)
            temp_results["Physical (Mask only)"]["cost"].append(cost)
            temp_results["Physical (Mask only)"]["fail_reasons"].extend(r_phy)

            sr, cost = calc_metrics(f_snr, c_snr)
            temp_results["Greedy-Comm (SNR)"]["sr"].append(sr)
            temp_results["Greedy-Comm (SNR)"]["cost"].append(cost)
            temp_results["Greedy-Comm (SNR)"]["fail_reasons"].extend(r_snr)

            sr, cost = calc_metrics(f_stab, c_stab)
            temp_results["Linear Prediction (Stability)"]["sr"].append(sr)
            temp_results["Linear Prediction (Stability)"]["cost"].append(cost)
            temp_results["Linear Prediction (Stability)"]["fail_reasons"].extend(r_stab)

        for scheme in results.keys():
            results[scheme]["success_rates"].append(np.mean(temp_results[scheme]["sr"]))
            results[scheme]["avg_costs"].append(np.mean(temp_results[scheme]["cost"]))
            results[scheme]["all_fail_reasons"].append(temp_results[scheme]["fail_reasons"])

    # ── 숫자 결과 테이블 ──
    FAIL_REASONS = ['none', 'deadline', 'connection', 'local_deadline', 'invalid_sv']
    print("\n" + "="*90)
    print(" [Numerical Results Table]")
    print("="*90)
    for ti, tr in enumerate(TURN_RATIOS):
        print(f"\nTurn Ratio: {tr}%")
        col_w = 10
        header = f"  {'Scheme':<32} | {'Success%':>8} | {'AvgCost':>8} | " + \
                 " | ".join(f"{r[:col_w]:>{col_w}}" for r in FAIL_REASONS)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for scheme in results.keys():
            sr   = results[scheme]["success_rates"][ti]
            cost = results[scheme]["avg_costs"][ti]
            reasons = results[scheme]["all_fail_reasons"][ti]
            counts = {r: reasons.count(r) for r in FAIL_REASONS}
            row = f"  {scheme:<32} | {sr:>7.1f}% | {cost:>8.3f} | " + \
                  " | ".join(f"{counts[r]:>{col_w}}" for r in FAIL_REASONS)
            print(row)
    print("="*90)

    print("\n📊 모든 훈련 완료! 논문용 그래프를 생성합니다...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = {
        "Proposed (Emb+Mask)":          "red",
        "Physical (Mask only)":          "green",
        "Greedy-Comm (SNR)":             "blue",
        "Linear Prediction (Stability)": "purple",
    }
    markers = {
        "Proposed (Emb+Mask)":          "o",
        "Physical (Mask only)":          "s",
        "Greedy-Comm (SNR)":             "^",
        "Linear Prediction (Stability)": "D",
    }

    x_labels = [f"{tr}%" for tr in TURN_RATIOS]

    for scheme, data in results.items():
        ax1.plot(x_labels, data["success_rates"], label=scheme,
                 color=colors[scheme], marker=markers[scheme], linewidth=2, markersize=8)

    ax1.set_title('(a) Offloading Success Rate vs. Turn Ratio', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Intersection Turn Ratio', fontsize=12)
    ax1.set_ylabel('Success Rate (%)', fontsize=12)
    ax1.grid(linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10)
    ax1.set_ylim(20, 80)

    for scheme, data in results.items():
        valid_costs = [c if c > 0 else np.nan for c in data["avg_costs"]]
        ax2.plot(x_labels, valid_costs, label=scheme,
                 color=colors[scheme], marker=markers[scheme], linewidth=2, markersize=8)

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
