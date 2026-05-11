"""
[Stage 2 검증 스크립트] A_stage2_RL_test1.py
인코더(Stage 1)가 완성된 상태에서, RL 환경(SumoV2XEnv)에 주입했을 때
의도대로 동작하는지 단계별로 검증합니다.

검증 항목:
  Step A: 인코더 로드 및 환경 초기화
  Step B: reset() 호출 → State, Mask, T_conn 값 검증
  Step C: Proposed vs Physical 환경의 마스킹 차이 비교
  Step D: step() 호출 → 채점(GT 기반) 정상 동작 확인
  Step E: 100 에피소드 자동 실행 → 통계 수집
"""

import numpy as np
import torch
import os

from A_v2x_master_module import (
    set_seed, V2XConfig, SumoV2XEnv, 
    pretrain_intention_encoder, start_sumo_traci
)

def run_stage2_test():
    sumo_cfg_path = "sumo_jtr_scenarios/turn_40.sumocfg"
    set_seed(42)

    # ==========================================
    # Step A: 인코더 로드 및 환경 초기화
    # ==========================================
    print("="*70)
    print("🔬 [Step A] Stage 1 인코더 로드 + Stage 2 환경 초기화")
    print("="*70)
    
    encoder, predictor, traj_predictor = pretrain_intention_encoder(sumo_cfg_path, num_steps=5000, seed=42)
    print("✅ 인코더/Predictor 로드 완료\n")

    # Proposed 환경 (임베딩 O, 마스킹 O)
    config_proposed = V2XConfig()
    config_proposed.use_masking = True
    config_proposed.use_embedding = True
    
    # Physical 환경 (임베딩 X, 마스킹 O)
    config_physical = V2XConfig()
    config_physical.use_masking = True
    config_physical.use_embedding = False

    env_proposed = SumoV2XEnv(sumo_cfg_path, config_proposed, encoder, predictor,
                               pretrained_trajectory_predictor=traj_predictor, sumo_seed=42)
    env_physical = SumoV2XEnv(sumo_cfg_path, config_physical, None, None,
                               pretrained_trajectory_predictor=None, sumo_seed=42)
    
    print(f"  State 차원: {env_proposed.state_dim} (2 + 10×3 = 32)")
    print(f"  Action 차원: {env_proposed.action_dim} (로컬 1 + SV 10 = 11)")
    print("✅ 두 환경 초기화 완료\n")

    # ==========================================
    # Step B: reset() 호출 → 내부 값 상세 출력
    # ==========================================
    print("="*70)
    print("🔬 [Step B] reset() 호출 → State, Mask, T_conn 상세 검증")
    print("="*70)
    
    try:
        traci.close()
    except:
        pass

    env_proposed.start_sumo()
    state, mask = env_proposed.reset()
    
    print(f"\n📋 태스크 정보:")
    print(f"  타입: {env_proposed.task['name']}")
    print(f"  데이터 크기(D): {env_proposed.task['D']:.1f}")
    print(f"  연산량(C): {env_proposed.task['C']:.1f}")
    print(f"  데드라인(T_max): {env_proposed.task['T_max']:.1f}초")
    
    print(f"\n📋 SV 정보 ({env_proposed.last_num_svs}대 발견):")
    print(f"  {'SV#':<5} {'CPU(GHz)':<10} {'R(Mbps)':<10} {'T_pred(s)':<12} {'T_gt(s)':<10} {'Mask':<6}")
    print(f"  {'-'*53}")
    
    for i in range(env_proposed.last_num_svs):
        cpu = env_proposed.f_sv[i]
        rate = env_proposed.R_sv[i]
        t_pred = env_proposed.T_conn_predicted[i]
        t_gt = env_proposed.T_conn_gt[i]
        m = int(mask[i+1])
        
        # 태스크 처리 시간 계산
        total_time = env_proposed.task['D'] / rate + env_proposed.task['C'] / cpu
        
        flag = ""
        if m == 0:
            if total_time > env_proposed.task['T_max']:
                flag = "← 데드라인 초과"
            elif total_time > t_pred:
                flag = "← T_conn 부족 (필터링됨!)"
        
        print(f"  SV{i:<3} {cpu:<10.1f} {rate:<10.1f} {t_pred:<12.1f} {t_gt:<10.1f} {m:<6} {flag}")
    
    local_time = env_proposed.task['C'] / config_proposed.f_tv
    print(f"\n  로컬 처리 시간: {local_time:.1f}초 (데드라인: {env_proposed.task['T_max']:.1f}초) → Mask[0]={int(mask[0])}")
    
    print(f"\n📋 State 벡터 (32차원):")
    print(f"  [0] D/100 = {state[0]:.3f} → D = {state[0]*100:.1f}")
    print(f"  [1] C/100 = {state[1]:.3f} → C = {state[1]*100:.1f}")
    print(f"  [2~11] f_sv/50 = {state[2:12]}")
    print(f"  [12~21] R_sv/100 = {state[12:22]}")
    print(f"  [22~31] T_conn/30 = {state[22:32]}")
    
    print(f"\n📋 Action Mask: {mask}")
    print(f"  유효 액션 수: {int(np.sum(mask))}개 (로컬 포함)")
    
    env_proposed.close_sumo()
    print("\n✅ Step B 완료\n")

    # ==========================================
    # Step C: Proposed vs Physical 마스킹 차이 비교
    # ==========================================
    print("="*70)
    print("🔬 [Step C] Proposed vs Physical 마스킹 차이 비교 (20 에피소드)")
    print("="*70)
    
    proposed_masks = []
    physical_masks = []
    proposed_filtered = 0  # Proposed가 추가로 필터링한 SV 수
    total_svs = 0
    
    # 같은 시드로 두 환경을 번갈아 실행
    env_proposed.start_sumo()
    for ep in range(20):
        state_p, mask_p = env_proposed.reset()
        proposed_masks.append(np.sum(mask_p))
    env_proposed.close_sumo()
    
    env_physical.start_sumo()
    for ep in range(20):
        state_ph, mask_ph = env_physical.reset()
        physical_masks.append(np.sum(mask_ph))
    env_physical.close_sumo()
    
    print(f"\n📋 20 에피소드 평균 유효 액션 수:")
    print(f"  Proposed (임베딩+마스킹): {np.mean(proposed_masks):.1f} ± {np.std(proposed_masks):.1f}")
    print(f"  Physical (마스킹만):      {np.mean(physical_masks):.1f} ± {np.std(physical_masks):.1f}")
    diff = np.mean(physical_masks) - np.mean(proposed_masks)
    print(f"  차이: {diff:.1f}개 (Proposed가 추가로 필터링한 SV)")
    
    if diff > 0:
        print(f"  → ✅ Proposed가 Physical보다 엄격하게 필터링 중 (정상)")
    elif diff == 0:
        print(f"  → ⚠️ 차이 없음: 인코더가 모든 SV를 안전하다고 판단 중")
    else:
        print(f"  → ❌ Proposed가 Physical보다 많이 허용 (비정상)")
    
    print("\n✅ Step C 완료\n")

    # ==========================================
    # Step D: step() 호출 → 채점 정상 동작 확인
    # ==========================================
    print("="*70)
    print("🔬 [Step D] step() 채점 검증 (각 액션별 결과 확인)")
    print("="*70)
    
    env_proposed.start_sumo()
    state, mask = env_proposed.reset()
    
    print(f"\n📋 태스크: {env_proposed.task['name']} (D={env_proposed.task['D']}, C={env_proposed.task['C']}, T_max={env_proposed.task['T_max']})")
    print(f"  SV 수: {env_proposed.last_num_svs}, 유효 액션: {int(np.sum(mask))}개\n")
    
    # 로컬 처리 (action=0) 시뮬레이션
    local_time = env_proposed.task['C'] / config_proposed.f_tv
    local_energy = config_proposed.kappa * env_proposed.task['C'] * (config_proposed.f_tv ** 2)
    local_cost = config_proposed.alpha * local_time + config_proposed.beta * local_energy
    local_fail = local_time > env_proposed.task['T_max']
    print(f"  [Action 0] 로컬 처리:")
    print(f"    시간={local_time:.2f}초, 에너지={local_energy:.4f}, 비용={local_cost:.2f}")
    print(f"    데드라인 {'초과 → 실패!' if local_fail else '이내 → 성공'}")
    
    # 각 SV 오프로딩 시뮬레이션
    for i in range(min(env_proposed.last_num_svs, 5)):  # 처음 5개 SV만 출력
        t_trans = env_proposed.task['D'] / env_proposed.R_sv[i]
        t_comp = env_proposed.task['C'] / env_proposed.f_sv[i]
        total = t_trans + t_comp
        e_trans = config_proposed.p_tx * t_trans
        cost = config_proposed.alpha * total + config_proposed.beta * e_trans
        
        fail_reason = ""
        if total > env_proposed.task['T_max']:
            fail_reason = "데드라인 초과"
        elif total > env_proposed.T_conn_gt[i]:
            fail_reason = f"GT T_conn({env_proposed.T_conn_gt[i]:.1f}s) 초과"
        
        status = f"실패({fail_reason})" if fail_reason else "성공"
        masked = "허용" if mask[i+1] == 1 else "차단"
        
        print(f"\n  [Action {i+1}] SV{i} 오프로딩 (마스크: {masked}):")
        print(f"    전송={t_trans:.2f}초 + 연산={t_comp:.2f}초 = {total:.2f}초")
        print(f"    T_conn_pred={env_proposed.T_conn_predicted[i]:.1f}초, T_conn_gt={env_proposed.T_conn_gt[i]:.1f}초")
        print(f"    비용={cost:.2f}, 결과: {status}")
    
    # 실제 step() 호출하여 reward 확인
    valid_actions = np.where(mask == 1.0)[0]
    if len(valid_actions) > 0:
        test_action = valid_actions[0]
        _, reward, _, info = env_proposed.step(test_action)
        print(f"\n  📌 실제 step(action={test_action}) 실행:")
        print(f"    reward={reward:.2f}, failed={info['failed']}, cost={info['cost']:.2f}")
    
    env_proposed.close_sumo()
    print("\n✅ Step D 완료\n")

    # ==========================================
    # Step E: 100 에피소드 자동 실행 → 통계 수집
    # ==========================================
    print("="*70)
    print("🔬 [Step E] 100 에피소드 자동 실행 → Proposed vs Physical 통계 비교")
    print("="*70)
    
    NUM_EPISODES = 100
    
    results = {}
    
    for scheme_name, config, enc, pred, traj_pred in [
        ("Proposed", config_proposed, encoder, predictor, traj_predictor),
        ("Physical", config_physical, None, None, None)
    ]:
        env = SumoV2XEnv(sumo_cfg_path, config, enc, pred,
                         pretrained_trajectory_predictor=traj_pred, sumo_seed=42)
        env.start_sumo()
        
        fails, costs, valid_acts, rewards_list = [], [], [], []
        zero_sv_count = 0
        
        for ep in range(NUM_EPISODES):
            state, mask = env.reset()
            
            # 유효한 액션 중 랜덤 선택 (학습 전이므로)
            valid = np.where(mask == 1.0)[0]
            action = np.random.choice(valid) if len(valid) > 0 else 0
            
            _, reward, _, info = env.step(action)
            
            fails.append(1 if info['failed'] else 0)
            costs.append(info['cost'])
            valid_acts.append(info['num_valid_actions'])
            rewards_list.append(reward)
            
            if info['num_svs'] == 0:
                zero_sv_count += 1
        
        env.close_sumo()
        
        fail_rate = np.mean(fails) * 100
        avg_cost_success = np.mean([c for c, f in zip(costs, fails) if f == 0]) if sum(fails) < NUM_EPISODES else float('nan')
        avg_valid = np.mean(valid_acts)
        avg_reward = np.mean(rewards_list)
        
        results[scheme_name] = {
            'fail_rate': fail_rate,
            'avg_cost': avg_cost_success,
            'avg_valid': avg_valid,
            'avg_reward': avg_reward
        }
        
        print(f"\n📊 {scheme_name} ({NUM_EPISODES} 에피소드, 랜덤 액션):")
        print(f"  실패율: {fail_rate:.1f}%")
        print(f"  평균 비용 (성공만): {avg_cost_success:.2f}")
        print(f"  평균 유효 액션 수: {avg_valid:.1f}")
        print(f"  평균 Reward: {avg_reward:.2f}")
        print(f"  SV 0대 에피소드: {zero_sv_count}회 ({zero_sv_count/NUM_EPISODES*100:.1f}%)")
    
    # 비교 분석
    print(f"\n{'='*50}")
    print(f"📊 Proposed vs Physical 비교 분석:")
    print(f"{'='*50}")
    print(f"  {'지표':<20} {'Proposed':<15} {'Physical':<15} {'차이':<15}")
    print(f"  {'-'*60}")
    
    for metric, label in [('fail_rate', '실패율(%)'), ('avg_cost', '평균비용'), 
                           ('avg_valid', '유효액션수'), ('avg_reward', '평균Reward')]:
        p = results['Proposed'][metric]
        ph = results['Physical'][metric]
        diff = p - ph
        print(f"  {label:<20} {p:<15.2f} {ph:<15.2f} {diff:+.2f}")
    
    if results['Proposed']['fail_rate'] < results['Physical']['fail_rate']:
        print(f"\n  → ✅ Proposed가 Physical보다 실패율이 낮음 (인코더 필터링 효과 확인)")
    else:
        print(f"\n  → ⚠️ Proposed가 Physical보다 실패율이 같거나 높음 (추가 분석 필요)")
    
    print(f"\n{'='*70}")
    print(f"🏁 Stage 2 검증 완료!")
    print(f"{'='*70}")
    print(f"  Step A: 인코더 로드 ✅")
    print(f"  Step B: reset() 내부 값 확인 ✅")
    print(f"  Step C: Proposed vs Physical 마스킹 차이 확인 ✅")
    print(f"  Step D: step() 채점 로직 확인 ✅")
    print(f"  Step E: 100 에피소드 통계 비교 ✅")
    print(f"\n  위 결과가 모두 정상이면 Stage 3(PPO 학습)으로 진입할 수 있습니다.")

if __name__ == "__main__":
    run_stage2_test()