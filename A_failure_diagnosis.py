"""
A_failure_diagnosis.py
======================
V2V 연구 실패 원인 정량 분석 스크립트.

스킴별로 실패가 (1)deadline, (2)connection, (3)local_deadline, (4)invalid_sv
중 어디서 발생하는지 breakdown하여 비교 표 + 스택 바 차트로 출력한다.

실행:
    python A_failure_diagnosis.py

결과 파일 저장 경로: /mnt/user-data/outputs/
"""

import os, sys, copy, time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# ─── 마스터 모듈 임포트 ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from A_v2x_master_module import (
    set_seed, V2XConfig, SumoV2XEnv, RolloutBuffer,
    FullPPOAgent, GreedySNRAgent, GreedyStabilityAgent,
    pretrain_intention_encoder, start_sumo_traci,
)
import traci

# ─── 설정 ────────────────────────────────────────────────────────────────────
SUMOCFG     = "sumo_jtr_scenarios/turn_40.sumocfg"
SEED        = 42
PRETRAIN_STEPS  = 8000
PPO_TRAIN_EPS   = 1500   # PPO 학습 에피소드 수
PPO_EVAL_EPS    = 300    # PPO 평가 에피소드 수
GREEDY_EVAL_EPS = 500    # Greedy 평가 에피소드 수
PPO_UPDATE_EVERY = 64    # 몇 에피소드마다 PPO 업데이트

OUTPUT_DIR  = OUTPUT_DIR = "/Users/eunseo/Desktop/NetworkLab/2.20test/failure_diagnosis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FAIL_REASONS = ['deadline', 'connection', 'local_deadline', 'invalid_sv']
TASK_TYPES   = ['CoopPerception', 'ObjectDetect', 'SensorFusion', 'HDMapUpdate', 'LightNotify']

# ─── 에피소드 실행 헬퍼 ──────────────────────────────────────────────────────

def run_episode(env, agent, deterministic=False):
    """
    단일 에피소드를 실행하고 (reward, info) 를 반환한다.
    PPO 에이전트는 (state, action, logprob, value, reward, done, mask) 를,
    Greedy 에이전트는 info 만 필요하다.
    """
    state, mask = env.reset()
    state_t = torch.FloatTensor(state).unsqueeze(0)
    mask_t  = torch.FloatTensor(mask).unsqueeze(0)

    action, logprob, value = agent.get_action(state_t, mask_t, deterministic=deterministic)
    next_state, reward, done, info = env.step(action.item())

    return reward, info, state, mask, action, logprob, value


def collect_stats(n_episodes, env, agent, deterministic=True):
    """n_episodes 동안 평가하여 실패 원인 통계 dict 반환."""
    stats = {
        'total': 0,
        'success': 0,
        'fail': 0,
        'reasons': defaultdict(int),          # reason -> count
        'task_reasons': defaultdict(lambda: defaultdict(int)),  # task_type -> reason -> count
        'task_total': defaultdict(int),
        'task_fail': defaultdict(int),
    }

    for ep in range(n_episodes):
        reward, info, *_ = run_episode(env, agent, deterministic=deterministic)
        tt = info['task_type']
        fr = info['fail_reason']

        stats['total'] += 1
        stats['task_total'][tt] += 1

        if info['failed']:
            stats['fail'] += 1
            stats['task_fail'][tt] += 1
            stats['reasons'][fr] += 1
            stats['task_reasons'][tt][fr] += 1
        else:
            stats['success'] += 1

        if (ep + 1) % 100 == 0:
            print(f"    [eval {ep+1}/{n_episodes}] success={stats['success']}, fail={stats['fail']}")

    return stats


# ─── PPO 학습 함수 ───────────────────────────────────────────────────────────

def train_ppo(env, agent, n_episodes, update_every=PPO_UPDATE_EVERY):
    """PPO 에이전트를 n_episodes 동안 학습."""
    buffer = RolloutBuffer()
    ep_rewards = []

    for ep in range(n_episodes):
        reward, info, state, mask, action, logprob, value = run_episode(env, agent, deterministic=False)

        buffer.states.append(state)
        buffer.actions.append(action.item())
        buffer.logprobs.append(logprob.detach() if logprob is not None else torch.tensor(0.0))
        buffer.rewards.append(reward)
        buffer.dones.append(True)
        buffer.masks.append(mask)
        buffer.values.append(value.item() if value is not None else 0.0)

        ep_rewards.append(reward)

        if (ep + 1) % update_every == 0:
            agent.update(buffer)
            buffer.clear()

        if (ep + 1) % 300 == 0:
            avg = np.mean(ep_rewards[-300:])
            print(f"    [train {ep+1}/{n_episodes}] avg_reward(last 300)={avg:.2f}")

    # 남은 버퍼 업데이트
    if len(buffer.states) > 0:
        agent.update(buffer)
        buffer.clear()

    print(f"    학습 완료. 최종 avg_reward={np.mean(ep_rewards[-300:]):.2f}")


# ─── 메인 ────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    print("=" * 70)
    print(" V2V 실패 원인 진단 스크립트 (A_failure_diagnosis.py)")
    print("=" * 70)

    # ── Phase 1: 인코더 사전 학습 ───────────────────────────────────────────
    print("\n[Phase 1] IntentionEncoder / ConnectionPredictor / TrajectoryPredictor 사전 학습")
    encoder, predictor, traj_predictor = pretrain_intention_encoder(
        SUMOCFG, num_steps=PRETRAIN_STEPS, seed=SEED
    )

    # ── 공통 설정 헬퍼 ──────────────────────────────────────────────────────
    def make_config(use_embedding):
        cfg = V2XConfig()
        cfg.use_masking  = True
        cfg.use_embedding = use_embedding
        return cfg

    def make_env(cfg, enc=None, pred=None, traj=None):
        env = SumoV2XEnv(
            SUMOCFG, cfg,
            pretrained_encoder=enc,
            pretrained_predictor=pred,
            pretrained_trajectory_predictor=traj,
            sumo_seed=SEED,
        )
        env.start_sumo()
        return env

    # ─────────────────────────────────────────────────────────────────────────
    # ── Scheme 1: Proposed (FullPPO + masking + embedding + traj) ────────────
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("[Scheme 1] Proposed — FullPPOAgent + masking + embedding + traj")
    cfg_prop = make_config(use_embedding=True)
    env_prop  = make_env(cfg_prop, encoder, predictor, traj_predictor)

    agent_prop = FullPPOAgent(env_prop.state_dim, env_prop.action_dim, cfg_prop)

    print(f"  [학습] {PPO_TRAIN_EPS} 에피소드 PPO 학습 중...")
    train_ppo(env_prop, agent_prop, PPO_TRAIN_EPS)

    print(f"  [평가] {PPO_EVAL_EPS} 에피소드 결정론적 평가 중...")
    stats_prop = collect_stats(PPO_EVAL_EPS, env_prop, agent_prop, deterministic=True)
    env_prop.close_sumo()

    # ─────────────────────────────────────────────────────────────────────────
    # ── Scheme 2: Physical (FullPPO + masking, NO embedding) ─────────────────
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("[Scheme 2] Physical — FullPPOAgent + masking, NO embedding")
    cfg_phys = make_config(use_embedding=False)
    env_phys  = make_env(cfg_phys)

    agent_phys = FullPPOAgent(env_phys.state_dim, env_phys.action_dim, cfg_phys)

    print(f"  [학습] {PPO_TRAIN_EPS} 에피소드 PPO 학습 중...")
    train_ppo(env_phys, agent_phys, PPO_TRAIN_EPS)

    print(f"  [평가] {PPO_EVAL_EPS} 에피소드 결정론적 평가 중...")
    stats_phys = collect_stats(PPO_EVAL_EPS, env_phys, agent_phys, deterministic=True)
    env_phys.close_sumo()

    # ─────────────────────────────────────────────────────────────────────────
    # ── Scheme 3: Greedy-SNR ─────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("[Scheme 3] Greedy-SNR — GreedySNRAgent + masking")
    cfg_snr  = make_config(use_embedding=False)
    env_snr   = make_env(cfg_snr)
    agent_snr = GreedySNRAgent(env_snr.state_dim, env_snr.action_dim, cfg_snr)

    print(f"  [평가] {GREEDY_EVAL_EPS} 에피소드 평가 중...")
    stats_snr = collect_stats(GREEDY_EVAL_EPS, env_snr, agent_snr, deterministic=True)
    env_snr.close_sumo()

    # ─────────────────────────────────────────────────────────────────────────
    # ── Scheme 4: Linear Prediction (GreedyStability) ────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("[Scheme 4] Linear Prediction — GreedyStabilityAgent + masking")
    cfg_lin  = make_config(use_embedding=False)
    env_lin   = make_env(cfg_lin)
    agent_lin = GreedyStabilityAgent(env_lin.state_dim, env_lin.action_dim, cfg_lin)

    print(f"  [평가] {GREEDY_EVAL_EPS} 에피소드 평가 중...")
    stats_lin = collect_stats(GREEDY_EVAL_EPS, env_lin, agent_lin, deterministic=True)
    env_lin.close_sumo()

    # ─────────────────────────────────────────────────────────────────────────
    # ── 결과 집계 및 출력 ────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    all_schemes = {
        'Proposed':           stats_prop,
        'Physical':           stats_phys,
        'Greedy-SNR':         stats_snr,
        'Linear Prediction':  stats_lin,
    }

    print("\n" + "=" * 70)
    print(" 실패 원인 분석 결과")
    print("=" * 70)

    # ── 1. 스킴별 요약 표 ─────────────────────────────────────────────────
    header = f"{'Scheme':<20} {'Total':>6} {'Success%':>9} {'Fail%':>7} " + \
             "  ".join(f"{r:>13}" for r in FAIL_REASONS)
    print("\n[스킴별 성공/실패율 및 실패 원인 Breakdown]")
    print(header)
    print("-" * len(header))

    for scheme, stats in all_schemes.items():
        total   = stats['total']
        success = stats['success']
        fail    = stats['fail']
        succ_pct = success / total * 100 if total > 0 else 0
        fail_pct = fail    / total * 100 if total > 0 else 0

        reason_parts = []
        for r in FAIL_REASONS:
            cnt = stats['reasons'][r]
            pct = cnt / total * 100 if total > 0 else 0
            reason_parts.append(f"{cnt:>5}({pct:>5.1f}%)")

        row = f"{scheme:<20} {total:>6} {succ_pct:>8.1f}% {fail_pct:>6.1f}%  " + \
              "  ".join(reason_parts)
        print(row)

    # ── 2. 태스크 타입별 실패 원인 분포 ──────────────────────────────────
    print("\n\n[태스크 타입별 실패 원인 분포]")
    for scheme, stats in all_schemes.items():
        print(f"\n  << {scheme} >>")
        sub_hdr = f"  {'Task':<10} {'Total':>6} {'Fail':>6} {'Fail%':>7}  " + \
                  "  ".join(f"{r:>13}" for r in FAIL_REASONS)
        print(sub_hdr)
        print("  " + "-" * (len(sub_hdr) - 2))
        for tt in TASK_TYPES:
            tt_total = stats['task_total'].get(tt, 0)
            tt_fail  = stats['task_fail'].get(tt, 0)
            if tt_total == 0:
                continue
            tt_fail_pct = tt_fail / tt_total * 100
            reason_parts = []
            for r in FAIL_REASONS:
                cnt = stats['task_reasons'][tt][r]
                pct = cnt / tt_total * 100
                reason_parts.append(f"{cnt:>5}({pct:>5.1f}%)")
            row = f"  {tt:<10} {tt_total:>6} {tt_fail:>6} {tt_fail_pct:>6.1f}%  " + \
                  "  ".join(reason_parts)
            print(row)

    # ── 3. 텍스트 파일로 저장 ─────────────────────────────────────────────
    txt_path = os.path.join(OUTPUT_DIR, "failure_diagnosis_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("V2V Failure Diagnosis Report\n")
        f.write("=" * 70 + "\n\n")

        f.write("[Scheme Summary]\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for scheme, stats in all_schemes.items():
            total   = stats['total']
            success = stats['success']
            fail    = stats['fail']
            succ_pct = success / total * 100 if total > 0 else 0
            fail_pct = fail    / total * 100 if total > 0 else 0
            reason_parts = []
            for r in FAIL_REASONS:
                cnt = stats['reasons'][r]
                pct = cnt / total * 100 if total > 0 else 0
                reason_parts.append(f"{cnt:>5}({pct:>5.1f}%)")
            row = f"{scheme:<20} {total:>6} {succ_pct:>8.1f}% {fail_pct:>6.1f}%  " + \
                  "  ".join(reason_parts)
            f.write(row + "\n")

        f.write("\n\n[Task-Type Breakdown]\n")
        for scheme, stats in all_schemes.items():
            f.write(f"\n<< {scheme} >>\n")
            for tt in TASK_TYPES:
                tt_total = stats['task_total'].get(tt, 0)
                tt_fail  = stats['task_fail'].get(tt, 0)
                if tt_total == 0:
                    continue
                tt_fail_pct = tt_fail / tt_total * 100
                reason_parts = []
                for r in FAIL_REASONS:
                    cnt = stats['task_reasons'][tt][r]
                    pct = cnt / tt_total * 100
                    reason_parts.append(f"{r}={cnt}({pct:.1f}%)")
                f.write(f"  {tt}: total={tt_total}, fail={tt_fail}({tt_fail_pct:.1f}%)  " +
                        ", ".join(reason_parts) + "\n")

    print(f"\n보고서 저장 완료: {txt_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # ── 4. 스택 바 차트 ──────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    scheme_names = list(all_schemes.keys())
    reason_colors = {
        'deadline':      '#E74C3C',   # 빨강
        'connection':    '#3498DB',   # 파랑
        'local_deadline':'#F39C12',   # 주황
        'invalid_sv':    '#9B59B6',   # 보라
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── (a) 절대 실패 수 스택 바 ─────────────────────────────────────────
    ax = axes[0]
    bottoms = np.zeros(len(scheme_names))
    for r in FAIL_REASONS:
        counts = [all_schemes[s]['reasons'][r] for s in scheme_names]
        ax.bar(scheme_names, counts, bottom=bottoms,
               color=reason_colors[r], label=r, edgecolor='white', linewidth=0.8)
        bottoms += np.array(counts, dtype=float)

    ax.set_title("Failure Count by Cause (absolute)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Scheme", fontsize=11)
    ax.set_ylabel("Number of Failures", fontsize=11)
    ax.legend(title="Fail Reason", bbox_to_anchor=(1.0, 1), loc='upper left', fontsize=9)
    ax.set_xticks(range(len(scheme_names)))
    ax.set_xticklabels(scheme_names, rotation=15, ha='right', fontsize=10)

    # 성공 수도 투명하게 표시
    success_counts = [all_schemes[s]['success'] for s in scheme_names]
    ax.bar(scheme_names, success_counts, bottom=bottoms,
           color='#2ECC71', alpha=0.3, label='success', edgecolor='white', linewidth=0.8)

    # ── (b) 실패 원인 비율 스택 바 (전체 대비 %) ──────────────────────────
    ax2 = axes[1]
    bottoms2 = np.zeros(len(scheme_names))
    for r in FAIL_REASONS:
        pcts = []
        for s in scheme_names:
            total = all_schemes[s]['total']
            cnt   = all_schemes[s]['reasons'][r]
            pcts.append(cnt / total * 100 if total > 0 else 0)
        ax2.bar(scheme_names, pcts, bottom=bottoms2,
                color=reason_colors[r], label=r, edgecolor='white', linewidth=0.8)
        bottoms2 += np.array(pcts)

    success_pcts = [all_schemes[s]['success'] / all_schemes[s]['total'] * 100 for s in scheme_names]
    ax2.bar(scheme_names, success_pcts, bottom=bottoms2,
            color='#2ECC71', alpha=0.35, label='success', edgecolor='white', linewidth=0.8)

    ax2.set_ylim(0, 105)
    ax2.set_title("Failure Breakdown (% of total episodes)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Scheme", fontsize=11)
    ax2.set_ylabel("% of Total Episodes", fontsize=11)
    ax2.legend(title="Fail Reason", bbox_to_anchor=(1.0, 1), loc='upper left', fontsize=9)
    ax2.set_xticks(range(len(scheme_names)))
    ax2.set_xticklabels(scheme_names, rotation=15, ha='right', fontsize=10)

    plt.suptitle("V2V Failure Cause Diagnosis — Scheme Comparison",
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    chart_path = os.path.join(OUTPUT_DIR, "failure_diagnosis_chart.png")
    plt.savefig(chart_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"차트 저장 완료: {chart_path}")

    # ── 태스크 타입별 서브플롯 (스킴 × 태스크 그리드) ─────────────────────
    fig2, axes2 = plt.subplots(1, len(all_schemes), figsize=(5 * len(all_schemes), 5), sharey=False)
    if len(all_schemes) == 1:
        axes2 = [axes2]

    for ax_i, (scheme, stats) in enumerate(all_schemes.items()):
        ax = axes2[ax_i]
        bottoms_t = np.zeros(len(TASK_TYPES))
        for r in FAIL_REASONS:
            counts_t = [stats['task_reasons'][tt][r] for tt in TASK_TYPES]
            ax.bar(TASK_TYPES, counts_t, bottom=bottoms_t,
                   color=reason_colors[r], label=r, edgecolor='white', linewidth=0.6)
            bottoms_t += np.array(counts_t, dtype=float)
        ax.set_title(scheme, fontsize=11, fontweight='bold')
        ax.set_xlabel("Task Type", fontsize=9)
        ax.set_ylabel("Failure Count", fontsize=9)
        ax.tick_params(axis='x', labelsize=8)
        if ax_i == len(all_schemes) - 1:
            ax.legend(title="Fail Reason", fontsize=7, loc='upper right')

    plt.suptitle("Task-Type × Failure Cause (per Scheme)",
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    chart2_path = os.path.join(OUTPUT_DIR, "failure_by_tasktype_chart.png")
    plt.savefig(chart2_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"태스크 타입별 차트 저장 완료: {chart2_path}")

    print("\n" + "=" * 70)
    print(" 모든 분석 완료.")
    print(f" 출력 파일 위치: {OUTPUT_DIR}/")
    print("   - failure_diagnosis_report.txt")
    print("   - failure_diagnosis_chart.png")
    print("   - failure_by_tasktype_chart.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
