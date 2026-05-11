"""
A_measure_disconnect_time.py
V2V 연구용 이탈 시간 분포 측정 스크립트

각 turn_ratio(10,25,40,55,70)에서 차량 쌍이 통신 범위(100m)를
벗어나기까지 걸리는 실제 시간을 측정하여 JSON으로 저장.
"""

import os
import math
import json
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
import traci

from A_v2x_master_module import start_sumo_traci, set_seed, SUMO_BINARY

COMM_RANGE  = 100.0
NUM_STEPS   = 5000
TURN_RATIOS = [10, 25, 40, 55, 70]
SCENARIO_DIR = "sumo_jtr_scenarios"
OUTPUT_JSON  = "disconnect_time_distribution.json"


# ──────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────

def _future_edges(vid, n=3):
    """차량의 현재 위치 이후 최대 n개 엣지 반환."""
    try:
        route = list(traci.vehicle.getRoute(vid))
        curr  = traci.vehicle.getRoadID(vid)
        if curr.startswith(':') or curr not in route:
            return route[:n]
        idx = route.index(curr)
        return route[idx : idx + n]
    except Exception:
        return []


def _common_edges(e1, e2):
    return len(set(e1) & set(e2))


def _stats(times):
    if not times:
        return {'mean': 0.0, 'median': 0.0, 'p25': 0.0, 'p75': 0.0, 'count': 0}
    arr = np.array(times, dtype=float)
    return {
        'mean':   float(np.mean(arr)),
        'median': float(np.median(arr)),
        'p25':    float(np.percentile(arr, 25)),
        'p75':    float(np.percentile(arr, 75)),
        'count':  int(len(arr)),
    }


# ──────────────────────────────────────────────
# 핵심 측정 함수
# ──────────────────────────────────────────────

def measure_disconnect_times(sumocfg_path, num_steps=NUM_STEPS):
    """
    단일 시나리오에서 이탈 시간 측정.

    Returns
    -------
    all_times       : list[float]  – 유효한 전체 연결 지속 시간
    diverging_times : list[float]  – 경로가 갈라지는 쌍 (common_edges == 0)
    same_path_times : list[float]  – 경로가 겹치는 쌍 (common_edges >= 1)
    """
    sumo_proc = start_sumo_traci(sumocfg_path)

    # (v1, v2) → {'start_t': float, 'common': int}
    active_pairs: dict = {}

    all_times, diverging_times, same_path_times = [], [], []

    try:
        for step in range(num_steps):
            traci.simulationStep()
            if traci.simulation.getMinExpectedNumber() <= 0:
                break

            veh_ids = traci.vehicle.getIDList()
            current_time = traci.simulation.getTime()
            pos_dict = {vid: traci.vehicle.getPosition(vid) for vid in veh_ids}

            # ── 이탈/소멸 쌍 처리 ──
            ended = []
            for (v1, v2), info in active_pairs.items():
                if v1 not in pos_dict or v2 not in pos_dict:
                    ended.append(((v1, v2), False))
                else:
                    p1, p2 = pos_dict[v1], pos_dict[v2]
                    dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                    if dist > COMM_RANGE:
                        ended.append(((v1, v2), True))

            for pair, is_valid in ended:
                info = active_pairs.pop(pair)
                if is_valid:
                    conn_t = current_time - info['start_t']
                    if conn_t > 1.0:          # 노이즈 제거: 1초 미만 무시
                        all_times.append(conn_t)
                        if info['common'] == 0:
                            diverging_times.append(conn_t)
                        else:
                            same_path_times.append(conn_t)

            # ── 새 쌍 등록 ──
            if len(veh_ids) < 4:
                continue

            sample_n = min(len(veh_ids), 20)
            sampled  = np.random.choice(veh_ids, sample_n, replace=False)

            for i in range(len(sampled)):
                for j in range(i + 1, len(sampled)):
                    v1, v2 = sampled[i], sampled[j]
                    if (v1, v2) in active_pairs or (v2, v1) in active_pairs:
                        continue
                    p1, p2 = pos_dict[v1], pos_dict[v2]
                    dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                    if dist <= COMM_RANGE:
                        fe1    = _future_edges(v1, n=3)
                        fe2    = _future_edges(v2, n=3)
                        common = _common_edges(fe1, fe2)
                        active_pairs[(v1, v2)] = {
                            'start_t': current_time,
                            'common':  common,
                        }

            if (step + 1) % 500 == 0:
                print(f"  step {step+1:>5}/{num_steps} | 활성 쌍: {len(active_pairs):>4} | "
                      f"기록: {len(all_times):>5} (갈라짐: {len(diverging_times)}, 같은경로: {len(same_path_times)})")

    finally:
        try:
            traci.close()
        except Exception:
            pass

    return all_times, diverging_times, same_path_times


# ──────────────────────────────────────────────
# 플롯 함수
# ──────────────────────────────────────────────

def _plot_all_histograms(results):
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle('V2V Disconnect Time Distribution by Turn Ratio', fontsize=14, fontweight='bold')

    for ax, ratio in zip(axes, TURN_RATIOS):
        key  = f"turn_{ratio}"
        data = results[key]['all_times']
        if data:
            ax.hist(data, bins=30, color='steelblue', edgecolor='white', alpha=0.85)
            mean_v = np.mean(data)
            ax.axvline(mean_v, color='red', linestyle='--', linewidth=1.5,
                       label=f'mean={mean_v:.1f}s')
            ax.legend(fontsize=8)
        ax.set_title(f'turn_{ratio}  (n={len(data)})', fontsize=10)
        ax.set_xlabel('Disconnect Time (s)')
        if ax is axes[0]:
            ax.set_ylabel('Count')

    plt.tight_layout()
    plt.savefig('disconnect_time_histograms.png', dpi=150)
    plt.show()
    print("[저장] disconnect_time_histograms.png")


def _plot_path_overlap(results):
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle('Disconnect Time: Diverging vs Same Path', fontsize=14, fontweight='bold')
    bins = np.linspace(0, 60, 31)

    for ax, ratio in zip(axes, TURN_RATIOS):
        key   = f"turn_{ratio}"
        div_t  = results[key]['diverging_times']
        same_t = results[key]['same_path_times']
        if div_t:
            ax.hist(div_t,  bins=bins, alpha=0.65, color='tomato',
                    label=f'Diverging (n={len(div_t)})',  density=True)
        if same_t:
            ax.hist(same_t, bins=bins, alpha=0.65, color='steelblue',
                    label=f'Same Path (n={len(same_t)})', density=True)
        ax.set_title(f'turn_{ratio}', fontsize=10)
        ax.set_xlabel('Disconnect Time (s)')
        if ax is axes[0]:
            ax.set_ylabel('Density')
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig('disconnect_time_by_path_overlap.png', dpi=150)
    plt.show()
    print("[저장] disconnect_time_by_path_overlap.png")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    set_seed(42)
    results = {}

    for ratio in TURN_RATIOS:
        cfg_path = os.path.join(SCENARIO_DIR, f"turn_{ratio}.sumocfg")
        print(f"\n{'='*60}")
        print(f"[turn_{ratio}] 측정 시작  ({NUM_STEPS} steps)")
        print(f"{'='*60}")

        all_t, div_t, same_t = measure_disconnect_times(cfg_path, NUM_STEPS)

        key = f"turn_{ratio}"
        results[key] = {
            'all_times':       all_t,
            'diverging_times': div_t,
            'same_path_times': same_t,
            'stats':           _stats(all_t),   # 전체 분포 요약 (JSON 스펙 준수)
            'stats_diverging': _stats(div_t),
            'stats_same_path': _stats(same_t),
        }

        s = results[key]['stats']
        print(f"  → 총 {s['count']}쌍 | 평균 {s['mean']:.2f}s | 중앙값 {s['median']:.2f}s")

    # ── JSON 저장 ──
    json_out = {}
    for key, val in results.items():
        json_out[key] = {
            'all_times':       val['all_times'],
            'diverging_times': val['diverging_times'],
            'same_path_times': val['same_path_times'],
            'stats':           val['stats'],
        }
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print(f"\n[저장] {OUTPUT_JSON}")

    # ── 콘솔 요약 테이블 ──
    header = (f"{'Turn Ratio':>10} | {'총 쌍':>6} | "
              f"{'평균(전체)':>10} | {'평균(갈라짐)':>12} | "
              f"{'평균(같은경로)':>13} | {'중앙값(갈라짐)':>14}")
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for ratio in TURN_RATIOS:
        key = f"turn_{ratio}"
        sa  = results[key]['stats']
        sd  = results[key]['stats_diverging']
        ss  = results[key]['stats_same_path']
        print(f"{ratio:>10} | {sa['count']:>6} | "
              f"{sa['mean']:>10.2f} | "
              f"{sd['mean']:>12.2f} | "
              f"{ss['mean']:>13.2f} | "
              f"{sd['median']:>14.2f}")
    print(sep)

    # ── 히스토그램 ──
    _plot_all_histograms(results)
    _plot_path_overlap(results)


if __name__ == "__main__":
    main()
