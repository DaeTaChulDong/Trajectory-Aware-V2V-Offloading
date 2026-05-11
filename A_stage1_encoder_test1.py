"""
[인코더 단독 테스트 스크립트]
Phase 2(RL)를 전혀 건드리지 않고, Phase 1(인코더+Predictor)만 학습 및 검증합니다.
교수님 피드백: "인코더가 90% 수준으로 동작하는지 먼저 확인한 뒤 다음 단계로"
"""

import numpy as np 
import torch
import matplotlib.pyplot as plt

# 마스터 모듈에서 Phase 1 관련 함수만 Import
from A_v2x_master_module import (
    set_seed, pretrain_intention_encoder
)

def run_encoder_test():
    # ==========================================
    # 설정
    # ==========================================
    # 🌟 여기에 본인의 sumocfg 경로를 넣으세요
    sumo_cfg_path = "sumo_jtr_scenarios/turn_40.sumocfg"
    
    set_seed(42)
    
    print("="*70)
    print("🧪 [인코더 단독 테스트] Phase 1만 실행하여 성능을 검증합니다.")
    print("   목표: 인코더가 연결 시간을 잘 예측하는지 확인")
    print("   기준: Binary 정확도 > 70%, 유사도 차이 > 0.2")
    print("="*70)
    
    # ==========================================
    # Phase 1 실행 (pretrain 내부에서 학습 + 검증까지 수행)
    # ==========================================
    encoder, predictor, traj_predictor = pretrain_intention_encoder(
        sumo_cfg_path,
        num_steps=5000,  # SUMO 시뮬레이션 step 수 (데이터 수집량에 영향)
        seed=42
    )
    
    print("\n✅ 인코더 단독 테스트 완료!")
    print("📊 위의 검증 결과를 확인하세요:")
    print("   1. 산점도(predictor_validation_scatter.png)에서 점들이 대각선에 모이는가?")
    print("   2. 유사도 차이가 0.2 이상인가?")
    print("   3. Binary 정확도가 70% 이상인가?")
    print("   4. 구간별 MAE가 합리적인가? (0~5초 구간이 가장 어려움)")
    print("\n   ↑ 이 기준을 만족하면 Phase 2(RL 환경)로 넘어갈 수 있습니다.")
    print("   ↑ 만족하지 못하면 인코더 구조/학습 데이터/에폭 수를 조정하세요.")

if __name__ == "__main__":
    run_encoder_test()
