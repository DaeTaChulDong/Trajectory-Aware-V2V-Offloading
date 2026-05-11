"""
A_measure_disconnect_time.py  
    측정:  |---- 같은 도로 ----|----- 갈라진 후 -----|
            0초              200초                 230초
            진입               분기점               이탈
            ← 총 230초 (total_connect_time_distribution.json에서 기록된 값) →

A_measure_diverge_to_disconnect.py
    측정:                    |--- 이 구간만 ---|
                              분기점             이탈
                              ← 30초 (진짜 필요한 값) →
"""
import os
import sys
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

from A_v2x_master_module import start_sumo_traci, set_seed, SUMO_BINARY
import sumolib
import traci

COMM_RANGE = 100.0
SIM_STEPS = 5000
TURN_RATIOS = [10, 25, 40, 55, 70]
SCENARIO_DIR = "sumo_jtr_scenarios"
OUTPUT_JSON = "diverge_disconnect_times.json"
OUTPUT_PNG = "diverge_disconnect_distribution.png"


def euclidean_dist(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def run_simulation(turn_ratio):
    cfg_path = os.path.join(SCENARIO_DIR, f"turn_{turn_ratio}.sumocfg")
    proc = start_sumo_traci(cfg_path, sumo_binary=SUMO_BINARY)

    # prev_edges[vid] = edge at previous step
    prev_edges = {}

    # active_pairs: dict of (v1, v2) -> diverge_step
    # 분기 이벤트가 발생하고 아직 이탈하지 않은 쌍
    active_pairs = {}

    times = []
    distances_at_diverge = []

    try:
        for step in range(SIM_STEPS):
            traci.simulationStep()

            current_vehicles = set(traci.vehicle.getIDList())
            if not current_vehicles:
                prev_edges = {}
                continue

            # 현재 step의 엣지와 위치 수집
            cur_edges = {}
            cur_pos = {}
            for vid in current_vehicles:
                cur_edges[vid] = traci.vehicle.getRoadID(vid)
                cur_pos[vid] = traci.vehicle.getPosition(vid)

            # 사라진 차량이 포함된 active_pairs 폐기
            to_remove = [pair for pair in active_pairs if pair[0] not in current_vehicles or pair[1] not in current_vehicles]
            for pair in to_remove:
                del active_pairs[pair]

            # 통신 범위 내 모든 차량 쌍 검사
            vehicle_list = list(current_vehicles)
            for i in range(len(vehicle_list)):
                for j in range(i + 1, len(vehicle_list)):
                    v1, v2 = vehicle_list[i], vehicle_list[j]
                    dist = euclidean_dist(cur_pos[v1], cur_pos[v2])
                    pair = (v1, v2)

                    if pair in active_pairs:
                        # 분기 이후 추적 중: 이탈 여부 확인
                        if dist > COMM_RANGE:
                            diverge_step = active_pairs[pair][0]
                            disconnect_time = (step - diverge_step) * traci.simulation.getDeltaT()
                            times.append(disconnect_time)
                            distances_at_diverge.append(active_pairs[pair][1])
                            del active_pairs[pair]
                    else:
                        # 분기 이벤트 감지 (100m 이내일 때만)
                        if dist <= COMM_RANGE:
                            v1_prev = prev_edges.get(v1)
                            v2_prev = prev_edges.get(v2)
                            if (v1_prev is not None and v2_prev is not None
                                    and v1_prev == v2_prev           # 이전 step: 같은 엣지
                                    and cur_edges[v1] != cur_edges[v2]  # 이번 step: 다른 엣지
                                    and cur_edges[v1] != v1_prev        # v1이 실제로 이동
                                    and cur_edges[v2] != v2_prev):      # v2도 실제로 이동
                                active_pairs[pair] = (step, dist)

            prev_edges = cur_edges

    finally:
        try:
            traci.close()
        except Exception:
            pass
        try:
            proc.terminate()
        except Exception:
            pass

    return times, distances_at_diverge


def compute_stats(times):
    if not times:
        return {"count": 0, "mean": None, "median": None,
                "p10": None, "p25": None, "p75": None, "p90": None}
    arr = np.array(times)
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def print_table(results):
    header = f"{'Turn%':>6} | {'분기 쌍 수':>10} | {'평균':>6} | {'중앙값':>6} | {'10%':>6} | {'25%':>6} | {'75%':>6} | {'90%':>6}"
    print(header)
    print("-" * len(header))
    for ratio in TURN_RATIOS:
        key = f"turn_{ratio}"
        s = results[key]["stats"]
        count = s["count"]
        if count == 0:
            print(f"{ratio:>6} | {count:>10} | {'N/A':>6} | {'N/A':>6} | {'N/A':>6} | {'N/A':>6} | {'N/A':>6} | {'N/A':>6}")
        else:
            print(f"{ratio:>6} | {count:>10} | {s['mean']:>6.2f} | {s['median']:>6.2f} | "
                  f"{s['p10']:>6.2f} | {s['p25']:>6.2f} | {s['p75']:>6.2f} | {s['p90']:>6.2f}")


def plot_results(results):
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, len(TURN_RATIOS), figure=fig, hspace=0.45, wspace=0.35)

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    # 상단: turn_ratio별 히스토그램
    for idx, ratio in enumerate(TURN_RATIOS):
        ax = fig.add_subplot(gs[0, idx])
        key = f"turn_{ratio}"
        times = results[key]["times"]
        if times:
            max_t = max(times)
            bins = np.arange(0, max_t + 2, 1.0)
            ax.hist(times, bins=bins, color=colors[idx], edgecolor="white", alpha=0.85)
        ax.set_title(f"Turn {ratio}%", fontsize=11, fontweight="bold")
        ax.set_xlabel("Diverge→Disconnect (s)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.grid(axis="y", alpha=0.4)

    # 하단: 전체 비교 박스플롯
    ax_box = fig.add_subplot(gs[1, :])
    data_for_box = [results[f"turn_{r}"]["times"] for r in TURN_RATIOS]
    labels = [f"Turn {r}%" for r in TURN_RATIOS]

    bp = ax_box.boxplot(
        [d if d else [0] for d in data_for_box],
        labels=labels,
        patch_artist=True,
        notch=False,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax_box.set_title("Diverge → Disconnect Time by Turn Ratio", fontsize=13, fontweight="bold")
    ax_box.set_xlabel("Turn Ratio", fontsize=11)
    ax_box.set_ylabel("Time (s)", fontsize=11)
    ax_box.grid(axis="y", alpha=0.4)

    fig.suptitle("V2V Communication Disconnection After Intersection Diverge", fontsize=14, fontweight="bold", y=1.01)
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\n그래프 저장 완료: {OUTPUT_PNG}")
    plt.close()


def main():
    set_seed(42)
    results = {}

    for ratio in TURN_RATIOS:
        print(f"\n[turn_{ratio}] 시뮬레이션 실행 중 ({SIM_STEPS} steps)...")
        times, dist_at_diverge = run_simulation(ratio)
        stats = compute_stats(times)
        key = f"turn_{ratio}"
        results[key] = {
            "times": times,
            "distances_at_diverge": dist_at_diverge,
            "stats": stats,
        }
        print(f"  → 분기 이벤트 {stats['count']}건 수집 완료")

    # JSON 저장
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nJSON 저장 완료: {OUTPUT_JSON}")

    # 콘솔 통계 테이블
    print("\n===== 분기→이탈 시간 통계 (단위: 초) =====")
    print_table(results)

    # 시각화
    plot_results(results)


if __name__ == "__main__":
    main()
