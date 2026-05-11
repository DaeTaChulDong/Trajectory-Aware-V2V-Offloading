"""
[Stage 3 검증 스크립트] A_stage3_PPO_test1.py
Stage 2(RL 환경)가 검증된 상태에서, PPO 에이전트를 학습시키고
3개 스킴(Proposed / Physical / Standard)의 성능을 비교합니다.

검증 항목:
  Step A: 인코더 로드 + 3개 스킴 환경 구축
  Step B: 소규모 학습 (1000 에피소드, 시드 1개) → 학습이 되는지 확인
  Step C: 3개 스킴 학습 곡선 비교
  Step D: 수렴 후 성능 테이블 출력
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from A_v2x_master_module import (
    set_seed, V2XConfig, SumoV2XEnv, RolloutBuffer,
    FullPPOAgent, VanillaPPOAgent,
    pretrain_intention_encoder, start_sumo_traci
)

def train_single_scheme(scheme_name, agent_class, use_masking, use_embedding,
                        sumo_cfg_path, encoder, predictor, traj_predictor, episodes, seed):
    """하나의 스킴을 학습하고 에피소드별 기록을 반환"""
    config = V2XConfig()
    config.use_masking = use_masking
    config.use_embedding = use_embedding

    enc = encoder if use_embedding else None
    pred = predictor if use_embedding else None
    traj_pred = traj_predictor if use_embedding else None

    env = SumoV2XEnv(sumo_cfg_path, config, enc, pred,
                     pretrained_trajectory_predictor=traj_pred, sumo_seed=seed)
    agent = agent_class(env.state_dim, env.action_dim, config)
    buffer = RolloutBuffer()

    reward_history = []
    fail_history = []
    cost_history = []
    UPDATE_TIMESTEP = 64

    env.start_sumo()

    try:
        for ep in tqdm(range(episodes), desc=f"{scheme_name}"):
            state, mask = env.reset()
            state_t = torch.FloatTensor(state).unsqueeze(0)
            mask_t = torch.FloatTensor(mask).unsqueeze(0)

            action, log_prob, value = agent.get_action(state_t, mask_t)
            _, reward, done, info = env.step(action.item())

            buffer.states.append(state)
            buffer.actions.append(action.item())
            buffer.logprobs.append(log_prob.squeeze())
            buffer.rewards.append(reward)
            buffer.values.append(value.squeeze())
            buffer.masks.append(mask)
            buffer.dones.append(done)

            reward_history.append(reward)
            fail_history.append(1 if info['failed'] else 0)
            cost_history.append(info['cost'])

            if (ep + 1) % UPDATE_TIMESTEP == 0:
                agent.update(buffer)
                buffer.clear()
    finally:
        env.close_sumo()

    return reward_history, fail_history, cost_history


def moving_average(data, window=100):
    """이동 평균 계산"""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def run_stage3_test():
    sumo_cfg_path = "sumo_jtr_scenarios/turn_40.sumocfg"
    set_seed(42)

    EPISODES = 1000  # 소규모 테스트 (정상 작동 확인용)
    SEED = 42

    # ==========================================
    # Step A: 인코더 로드
    # ==========================================
    print("="*70)
    print("🔬 [Step A] 인코더 로드")
    print("="*70)

    encoder, predictor, traj_predictor = pretrain_intention_encoder(sumo_cfg_path, num_steps=5000, seed=42)
    print("✅ 인코더 로드 완료\n")

    # ==========================================
    # Step B: 3개 스킴 소규모 학습
    # ==========================================
    print("="*70)
    print(f"🔬 [Step B] 3개 스킴 소규모 학습 ({EPISODES} 에피소드, 시드 {SEED})")
    print("="*70)

    schemes = [
        ("Proposed (Emb+Mask)", FullPPOAgent, True, True),
        ("Physical (Mask only)", FullPPOAgent, True, False),
        ("Standard (No Mask)",   VanillaPPOAgent, False, False),
    ]

    results = {}

    for name, agent_cls, use_mask, use_emb in schemes:
        print(f"\n▶ {name} 학습 시작...")
        set_seed(SEED)

        rewards, fails, costs = train_single_scheme(
            name, agent_cls, use_mask, use_emb,
            sumo_cfg_path, encoder, predictor, traj_predictor, EPISODES, SEED
        )

        results[name] = {
            'rewards': rewards,
            'fails': fails,
            'costs': costs
        }

        # 즉시 요약 출력
        last_200 = fails[-200:]
        print(f"  → 마지막 200 에피소드 실패율: {np.mean(last_200)*100:.1f}%")
        print(f"  → 마지막 200 에피소드 평균 Reward: {np.mean(rewards[-200:]):.2f}")

    # ==========================================
    # Step C: 학습 곡선 시각화
    # ==========================================
    print("\n" + "="*70)
    print("🔬 [Step C] 학습 곡선 비교 그래프 생성")
    print("="*70)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    plt.subplots_adjust(hspace=0.35)

    colors = {
        "Proposed (Emb+Mask)": "red",
        "Physical (Mask only)": "green",
        "Standard (No Mask)": "blue"
    }

    # (a) Moving Average Reward
    for name, data in results.items():
        ma = moving_average(data['rewards'], window=100)
        ax1.plot(range(len(ma)), ma, label=name, color=colors[name], linewidth=2)
    ax1.set_title('(a) Moving Average Reward (window=100)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Reward (Higher is Better)')
    ax1.legend()
    ax1.grid(linestyle='--', alpha=0.6)

    # (b) Cumulative Failures
    for name, data in results.items():
        cum_fails = np.cumsum(data['fails'])
        ax2.plot(range(len(cum_fails)), cum_fails, label=name, color=colors[name], linewidth=2)
    ax2.set_title('(b) Cumulative Failures', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Total Failures')
    ax2.legend()
    ax2.grid(linestyle='--', alpha=0.6)

    # (c) Failure Rate per Window
    window = 100
    for name, data in results.items():
        rates = [np.mean(data['fails'][i:i+window])*100
                 for i in range(len(data['fails'])-window+1)]
        ax3.plot(range(len(rates)), rates, label=name, color=colors[name], linewidth=2)
    ax3.set_title(f'(c) Failure Rate per Window ({window} Episodes)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Episodes')
    ax3.set_ylabel('Failure Rate (%)')
    ax3.legend()
    ax3.grid(linestyle='--', alpha=0.6)

    plt.savefig('stage3_learning_curves.png', dpi=300, bbox_inches='tight')
    print("✅ 학습 곡선 저장: 'stage3_learning_curves.png'")

    # ==========================================
    # Step D: 수렴 후 성능 테이블
    # ==========================================
    print("\n" + "="*70)
    print("🔬 [Step D] 수렴 후 성능 비교 (마지막 200 에피소드)")
    print("="*70)

    eval_window = 200

    print(f"\n  {'스킴':<25} {'실패율(%)':<12} {'평균Reward':<14} {'평균비용(성공)':<14}")
    print(f"  {'-'*65}")

    for name, data in results.items():
        last_fails = data['fails'][-eval_window:]
        last_rewards = data['rewards'][-eval_window:]
        last_costs = data['costs'][-eval_window:]
        last_fails_np = np.array(last_fails)
        last_costs_np = np.array(last_costs)

        fail_rate = np.mean(last_fails) * 100
        avg_reward = np.mean(last_rewards)
        success_costs = last_costs_np[last_fails_np == 0]
        avg_cost = np.mean(success_costs) if len(success_costs) > 0 else float('nan')

        print(f"  {name:<25} {fail_rate:<12.1f} {avg_reward:<14.2f} {avg_cost:<14.2f}")

    # 핵심 판단 기준 출력
    proposed_fail = np.mean(results["Proposed (Emb+Mask)"]['fails'][-eval_window:]) * 100
    physical_fail = np.mean(results["Physical (Mask only)"]['fails'][-eval_window:]) * 100
    standard_fail = np.mean(results["Standard (No Mask)"]['fails'][-eval_window:]) * 100

    print(f"\n📊 핵심 판단:")
    print(f"  Standard vs Physical 차이: {standard_fail - physical_fail:.1f}%p → ", end="")
    if standard_fail > physical_fail + 5:
        print("✅ 마스킹 효과 확인")
    else:
        print("⚠️ 마스킹 효과 미미")

    print(f"  Physical vs Proposed 차이: {physical_fail - proposed_fail:.1f}%p → ", end="")
    if physical_fail > proposed_fail + 3:
        print("✅ 임베딩 효과 확인")
    else:
        print("⚠️ 임베딩 효과 미미 (학습 에피소드 부족 가능)")

    # 학습 추세 확인 (reward가 우상향하는지)
    proposed_rewards = results["Proposed (Emb+Mask)"]['rewards']
    first_100 = np.mean(proposed_rewards[:100])
    last_100 = np.mean(proposed_rewards[-100:])
    improvement = last_100 - first_100

    print(f"\n📊 학습 추세 (Proposed):")
    print(f"  처음 100 에피소드 평균 Reward: {first_100:.2f}")
    print(f"  마지막 100 에피소드 평균 Reward: {last_100:.2f}")
    print(f"  개선: {improvement:+.2f} → ", end="")
    if improvement > 5:
        print("✅ PPO 학습이 진행되고 있음")
    elif improvement > 0:
        print("⚠️ 미미한 개선 (에피소드 수 부족 가능)")
    else:
        print("❌ 학습 안 됨 (환경/하이퍼파라미터 점검 필요)")

    print(f"\n{'='*70}")
    print(f"🏁 Stage 3 소규모 테스트 완료!")
    print(f"{'='*70}")
    print(f"  위 결과가 정상이면 대규모 실험으로 진입:")
    print(f"  → 5 seeds × 5000 episodes (실험 1)")
    print(f"  → 5개 회전 비율 × 3 seeds × 2000 episodes (실험 2)")


if __name__ == "__main__":
    run_stage3_test()