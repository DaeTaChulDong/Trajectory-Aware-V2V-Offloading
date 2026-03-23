import numpy as np
import matplotlib.pyplot as plt

def run_experiment_5_overhead():
    # 실험 변수: RSU 커버리지 내의 차량 밀도 (10대 ~ 100대)
    vehicle_densities = np.arange(10, 101, 10)
    
    # ==========================================
    # 1. 4가지 전략 파라미터 세팅 (현실적인 V2X 환경 모사)
    # ==========================================
    # [전략 1] Continuous-Raw (기존 1): 10Hz 주기로 날것의 궤적 좌표 계속 전송
    freq_cont_raw = 10.0      # 초당 10회
    size_cont_raw = 10.0      # 메시지당 10 KB (방대한 좌표 데이터)
    
    # [전략 2] Continuous-Embed: 10Hz 주기로 압축된 임베딩 벡터 전송
    freq_cont_emb = 10.0      # 초당 10회
    size_cont_emb = 0.5       # 메시지당 0.5 KB (가벼운 잠재 벡터)
    
    # [전략 3] Event-Raw: 교차로 접근 이벤트 시점에만 미래 궤적 세트 전송
    freq_event_raw = 0.5      # 초당 0.5회 (평균 2초마다 이벤트 발생)
    size_event_raw = 50.0     # 메시지당 50 KB (교차로 통과를 위한 무거운 전체 시퀀스)
    
    # [전략 4] Proposed (Event-Embed): 교차로 접근 시점에 압축된 의도(Intention) 벡터 전송
    freq_event_emb = 0.5      # 초당 0.5회
    size_event_emb = 1.0      # 메시지당 1.0 KB (임베딩 벡터)
    
    # CSMA/CA 충돌 계수: 전송 빈도와 차량 밀도가 높을수록 패킷 충돌로 인한 재전송 발생
    def calculate_collision_factor(freq, density):
        return 1.0 + (0.0005 * freq * density)
    
    # 에너지 소모 계수 (Joules per KB)
    energy_per_kb = 0.002 
    
    # 결과 저장용 리스트
    overhead_cr, overhead_ce, overhead_er, overhead_prop = [], [], [], []
    energy_cr, energy_ce, energy_er, energy_prop = [], [], [], []
    
    print("📡 [실험 5] 브로드캐스팅 및 임베딩 전략에 따른 네트워크 오버헤드 분석 시작...")
    
    # ==========================================
    # 2. 오버헤드 및 에너지 계산 로직
    # ==========================================
    for d in vehicle_densities:
        # [전략 1] Continuous-Raw
        cf_cr = calculate_collision_factor(freq_cont_raw, d)
        kb_per_sec_cr = d * freq_cont_raw * size_cont_raw * cf_cr
        overhead_cr.append(kb_per_sec_cr / 1024) # MB/s 변환
        energy_cr.append(kb_per_sec_cr * energy_per_kb * cf_cr) # 충돌 시 에너지 낭비 가중
        
        # [전략 2] Continuous-Embed
        cf_ce = calculate_collision_factor(freq_cont_emb, d)
        kb_per_sec_ce = d * freq_cont_emb * size_cont_emb * cf_ce
        overhead_ce.append(kb_per_sec_ce / 1024)
        energy_ce.append(kb_per_sec_ce * energy_per_kb * cf_ce)
        
        # [전략 3] Event-Raw
        cf_er = calculate_collision_factor(freq_event_raw, d)
        kb_per_sec_er = d * freq_event_raw * size_event_raw * cf_er
        overhead_er.append(kb_per_sec_er / 1024)
        energy_er.append(kb_per_sec_er * energy_per_kb * cf_er)
        
        # [전략 4] Proposed (Event-Embed)
        cf_prop = calculate_collision_factor(freq_event_emb, d)
        kb_per_sec_prop = d * freq_event_emb * size_event_emb * cf_prop
        overhead_prop.append(kb_per_sec_prop / 1024)
        energy_prop.append(kb_per_sec_prop * energy_per_kb * cf_prop)

    # ==========================================
    # 3. 듀얼 플롯 시각화 (논문 방어용)
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # [왼쪽] Signaling Overhead (MB/s)
    ax1.plot(vehicle_densities, overhead_cr, marker='x', markersize=8, color='gray', linestyle=':', linewidth=2, label='Continuous-Raw (10Hz, Uncompressed)')
    ax1.plot(vehicle_densities, overhead_er, marker='s', markersize=8, color='green', linestyle='--', linewidth=2.5, label='Event-Raw (0.5Hz, Trajectory Seq)')
    ax1.plot(vehicle_densities, overhead_ce, marker='^', markersize=8, color='blue', linestyle='-.', linewidth=2.5, label='Continuous-Embed (10Hz, Compressed)')
    ax1.plot(vehicle_densities, overhead_prop, marker='o', markersize=10, color='red', linewidth=3, label='Proposed Event-Embed (0.5Hz, Intention)')
    
    ax1.set_title('Network Signaling Overhead vs Vehicle Density', fontsize=14)
    ax1.set_xlabel('Number of Vehicles in RSU Coverage', fontsize=12)
    ax1.set_ylabel('Total Signaling Overhead (MB/s)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # [오른쪽] Energy Efficiency (Joules/s)
    ax2.plot(vehicle_densities, energy_cr, marker='x', markersize=8, color='gray', linestyle=':', linewidth=2, label='Continuous-Raw')
    ax2.plot(vehicle_densities, energy_er, marker='s', markersize=8, color='green', linestyle='--', linewidth=2.5, label='Event-Raw')
    ax2.plot(vehicle_densities, energy_ce, marker='^', markersize=8, color='blue', linestyle='-.', linewidth=2.5, label='Continuous-Embed')
    ax2.plot(vehicle_densities, energy_prop, marker='o', markersize=10, color='red', linewidth=3, label='Proposed Event-Embed')
    
    ax2.set_title('Communication Energy Consumption vs Density', fontsize=14)
    ax2.set_xlabel('Number of Vehicles in RSU Coverage', fontsize=12)
    ax2.set_ylabel('Total Energy Consumption (Joules/s)', fontsize=12)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.suptitle('Experiment 5: Efficiency of Event-Driven Broadcasting & Embedding Strategy', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('exp5_overhead_efficiency.png', dpi=300)
    print("\n✅ 완벽한 방어 논리! 실험 5 오버헤드 분석 완료. 'exp5_overhead_efficiency.png'가 저장되었습니다.")
    plt.show()

if __name__ == "__main__":
    run_experiment_5_overhead()