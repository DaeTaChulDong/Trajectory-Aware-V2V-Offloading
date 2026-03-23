import os
import sys
import traci
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("환경 변수 'SUMO_HOME'을 설정해주세요.")

def run_sumo_experiment3():
    sumo_cmd = ["sumo", "-c", "sumo_data/sim.sumocfg", "--no-warnings"]
    
    # 태스크 크기 배수 (1.0배 ~ 5.0배)
    task_scales = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    cost_proposed_list = []
    cost_local_list = []
    cost_greedy_list = []
    
    avg_t_conn_chosen_list = [] # Proposed가 선택한 차량의 평균 예상 연결 시간 (오른쪽 Y축용)
    
    T_max = 5.0 # 태스크가 커지므로 데드라인도 넉넉하게 확장
    tau_guard = 0.2
    T_error = 0.5
    comm_range = 150.0

    print("🚗 SUMO 실험 3: 태스크 크기 변화에 따른 동적 자원 선택 분석 시작...")
    
    for scale in task_scales:
        traci.start(sumo_cmd)
        
        costs_prop = []
        costs_loc = []
        costs_gre = []
        chosen_t_conns = []
        
        step = 0
        while step < 500: # 500초씩 빠르게 테스트
            traci.simulationStep()
            veh_ids = traci.vehicle.getIDList()
            
            if len(veh_ids) > 5 and step % 2 == 0:
                tv_id = np.random.choice(veh_ids)
                tv_pos = np.array(traci.vehicle.getPosition(tv_id))
                
                sv_candidates = []
                for vid in veh_ids:
                    if vid != tv_id:
                        pos = np.array(traci.vehicle.getPosition(vid))
                        dist = np.linalg.norm(tv_pos - pos)
                        if dist <= comm_range:
                            sv_candidates.append(vid)
                            
                num_svs = len(sv_candidates)
                if num_svs > 0:
                    # 🌟 태스크 크기를 scale에 비례하여 증가시킴
                    D_i = np.random.uniform(1.0, 2.0) * scale 
                    C_i = np.random.uniform(0.5, 1.0) * scale 
                    
                    R_sv = np.random.uniform(10.0, 30.0, num_svs)
                    f_sv = np.random.uniform(1.0, 4.0, num_svs)
                    
                    t_trans = D_i / R_sv
                    t_comp = C_i / f_sv
                    t_total = t_trans + t_comp
                    
                    # 실제 연결 시간 및 제안 기법의 예측 시간 (현실 반영 노이즈 추가)
                    is_turning = np.random.rand(num_svs) < 0.2 # 20% 턴 비율 고정
                    t_conn_actual = np.where(is_turning, np.random.uniform(0.5, 2.0, num_svs), np.random.uniform(4.0, 8.0, num_svs))
                    t_conn_proposed = t_conn_actual - np.random.uniform(0.0, 0.2, num_svs)
                    
                    # 비용 계산 (정규화)
                    E_local_base = (1.0 ** 2) * C_i
                    costs_all_svs = 0.9 * (t_total / T_max) + 0.1 * (((0.5 * t_trans) + (f_sv**2 * C_i) + 0.1) / E_local_base)
                    
                    # [기법 1] Local 단독 처리
                    t_local = C_i / 1.0
                    cost_local_val = 0.9 * (t_local / T_max) + 0.1 * (E_local_base / E_local_base)
                    # 데드라인 초과 시 패널티 부과
                    if t_local > T_max: cost_local_val = 5.0 
                    costs_loc.append(cost_local_val)
                    
                    # [기법 2] Greedy-CPU (가용 CPU가 제일 높은 놈 무조건 선택)
                    idx_greedy = np.argmax(f_sv)
                    if t_total[idx_greedy] <= T_max and t_total[idx_greedy] <= t_conn_actual[idx_greedy]:
                        costs_gre.append(costs_all_svs[idx_greedy])
                    else:
                        costs_gre.append(5.0) # 실패 시 큰 패널티 비용
                        
                    # [기법 3] Proposed (안전 마진 필터링 후 최소 비용 선택)
                    mask_proposed = (t_total <= T_max - tau_guard) & (t_total <= t_conn_proposed - T_error)
                    if np.any(mask_proposed):
                        valid_costs = np.where(mask_proposed, costs_all_svs, np.inf)
                        idx_proposed = np.argmin(valid_costs)
                        if t_total[idx_proposed] <= t_conn_actual[idx_proposed]:
                            costs_prop.append(costs_all_svs[idx_proposed])
                            # 🌟 선택한 놈의 예상 연결 시간을 기록 (오른쪽 Y축을 위해!)
                            chosen_t_conns.append(t_conn_proposed[idx_proposed])
                        else:
                            costs_prop.append(5.0)
                    else:
                        costs_prop.append(cost_local_val) # 안전한 놈 없으면 로컬 처리

        traci.close()
        
        cost_proposed_list.append(np.mean(costs_prop))
        cost_local_list.append(np.mean(costs_loc))
        cost_greedy_list.append(np.mean(costs_gre))
        avg_t_conn_chosen_list.append(np.mean(chosen_t_conns) if len(chosen_t_conns) > 0 else 0)
        
        print(f"Task Scale x{scale} | Proposed Cost: {cost_proposed_list[-1]:.2f} | Local Cost: {cost_local_list[-1]:.2f} | Greedy-CPU Cost: {cost_greedy_list[-1]:.2f} | 선택된 평균 T_conn: {avg_t_conn_chosen_list[-1]:.2f}초")

    # ==========================================
    # 이중 Y축 (Dual Y-axis) 그래프 그리기
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 왼쪽 Y축: 비용 (막대 그래프)
    bar_width = 0.25
    x = np.arange(len(task_scales))
    
    ax1.bar(x - bar_width, cost_proposed_list, bar_width, label='Proposed Cost', color='red', alpha=0.7)
    ax1.bar(x, cost_greedy_list, bar_width, label='Greedy-CPU Cost', color='blue', alpha=0.7)
    ax1.bar(x + bar_width, cost_local_list, bar_width, label='Local Only Cost', color='gray', alpha=0.7)
    
    ax1.set_xlabel('Task Size Multiplier (x)', fontsize=12)
    ax1.set_ylabel('Average Total Cost (Lower is Better)', fontsize=12, color='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"x{s}" for s in task_scales])
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim(0, 5.5)
    
    # 오른쪽 Y축: 에이전트가 선택한 차량의 평균 연결 지속 시간 (꺾은선 그래프)
    ax2 = ax1.twinx()
    ax2.plot(x, avg_t_conn_chosen_list, color='green', marker='o', markersize=10, linewidth=3, label='Chosen SV Avg Connection Time ($\hat{T}_{conn}$)')
    ax2.set_ylabel('Avg Connection Time of Chosen SV (sec)', fontsize=12, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim(0, 8.0)
    
    plt.title('Experiment 3: Dynamic Adaptation to Task Sizes', fontsize=16)
    
    # 두 축의 범례 합치기
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=10)
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('exp3_sumo_dynamic_adaptation.png', dpi=300)
    print("\n✅ 실험 3 완료! 'exp3_sumo_dynamic_adaptation.png'가 저장되었습니다.")
    plt.show()

if __name__ == "__main__":
    run_sumo_experiment3()