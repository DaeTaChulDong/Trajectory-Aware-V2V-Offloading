import gymnasium as gym
from gymnasium import spaces
import numpy as np

class V2VOffloadingEnv(gym.Env):
    """
    V2V 태스크 오프로딩을 위한 커스텀 강화학습 환경
    """
    def __init__(self, num_svs=10):
        super(V2VOffloadingEnv, self).__init__()
        
        # 시스템 파라미터 (논문의 설정값 반영)
        self.num_svs = num_svs           # 통신 반경 내 SV 후보 개수 (N)
        self.T_max = 2.0                 # 최대 허용 지연 시간 (초)
        self.tau_guard = 0.2             # 데드라인 가드 타임 (초) - C2 제약
        self.T_error = 0.5               # 임베딩 예측 오차 마진 (초) - C3 제약
        self.omega_t = 0.9               # 지연 시간 가중치
        self.omega_e = 0.1               # 에너지 가중치
        
        # 행동 공간 (Action Space): 0 (Local), 1 ~ N (SVs)
        self.action_space = spaces.Discrete(self.num_svs + 1)
        
        # 상태 공간 (Observation Space): [D_i, C_i, I_zone, f_sv(N개), R_sv(N개), T_conn(N개)]
        obs_dim = 3 + 3 * self.num_svs
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. 상태(State) 무작위 생성 (매 에피소드마다 새로운 상황 부여)
        self.D_i = np.random.uniform(1.0, 5.0)       # 데이터 크기 (Mb)
        self.C_i = np.random.uniform(0.5, 2.0)       # 요구 연산량 (Gcycles)
        self.I_zone = np.random.choice([0, 1])       # 1: 교차로(이벤트), 0: 일반 주행
        
        self.f_sv = np.random.uniform(1.0, 3.0, self.num_svs)    # SV들의 가용 CPU (GHz)
        self.R_sv = np.random.uniform(5.0, 20.0, self.num_svs)   # SV들의 전송 속도 (Mbps)
        
        # 임베딩 유사도에 따른 예상 연결 시간 (0.5초 ~ 4.0초)
        self.T_conn = np.random.uniform(0.5, 4.0, self.num_svs)  
        
        self.state = self._get_obs()
        
        # 2. 이번 턴에서 가능한 안전한 행동(Valid Actions) 마스크 추출
        action_mask = self._get_action_mask()
        
        return self.state, {"action_mask": action_mask}

    def _get_obs(self):
        """현재 상태를 1차원 numpy 배열로 반환"""
        obs = np.concatenate([
            [self.D_i, self.C_i, self.I_zone], 
            self.f_sv, self.R_sv, self.T_conn
        ])
        return obs.astype(np.float32)

    def _get_action_mask(self):
        """
        Stage 1: 물리적 제약 조건(C2, C3)을 기반으로 Action Mask 생성
        가능한 행동은 1, 불가능한 행동은 0으로 반환
        """
        mask = np.zeros(self.num_svs + 1, dtype=np.int8)
        
        # Action 0 (Local Processing) 검사
        f_local = 1.0 # 로컬 CPU 가정
        t_local = self.C_i / f_local
        if t_local <= self.T_max - self.tau_guard:
            mask[0] = 1
            
        # Action 1~N (V2V Offloading) 검사
        for j in range(self.num_svs):
            t_trans = self.D_i / self.R_sv[j]
            t_comp = self.C_i / self.f_sv[j]
            t_total = t_trans + t_comp
            
            # C2 (데드라인) 및 C3 (안전 마진 포함 연결 시간) 검사
            if (t_total <= self.T_max - self.tau_guard) and \
               (t_total <= self.T_conn[j] - self.T_error):
                mask[j + 1] = 1
                
        # 만약 조건을 만족하는 놈이 아무도 없으면 강제로 로컬(0) 허용 (오류 방지용)
        if np.sum(mask) == 0:
            mask[0] = 1
            
        return mask

    def step(self, action):
        """에이전트가 행동을 선택했을 때 환경의 변화와 보상을 계산"""
        # 로컬 처리 연산량 및 베이스라인 설정
        f_local = 1.0 
        E_local_base = (f_local ** 2) * self.C_i # 단순화된 에너지 계산
        
        # 이벤트 기반 오버헤드 (교차로 접근 시 통신 낭비 발생)
        E_overhead = 0.5 if self.I_zone == 1 else 0.1 

        t_total = 0
        e_total = 0
        is_failed = False
        penalty = 0

        if action == 0:
            # 로컬 처리
            t_total = self.C_i / f_local
            e_total = E_local_base
            if t_total > self.T_max - self.tau_guard:
                is_failed = True
                penalty = 100 * (t_total - self.T_max)
                
        else:
            # SV 오프로딩 (action 1 ~ N)
            idx = action - 1
            t_trans = self.D_i / self.R_sv[idx]
            t_comp = self.C_i / self.f_sv[idx]
            t_total = t_trans + t_comp
            
            e_trans = 0.5 * t_trans # 임의의 전송 전력 상수 가정
            e_comp = (self.f_sv[idx] ** 2) * self.C_i
            e_total = e_trans + e_comp
            
            # 제약조건(C2, C3) 위반 체크
            if t_total > self.T_max - self.tau_guard:
                is_failed = True
                penalty += 100 * (t_total - self.T_max)
            if t_total > self.T_conn[idx] - self.T_error:
                is_failed = True
                penalty += 100 * (t_total - self.T_conn[idx])

        # 비용(Cost) 계산 (정규화된 가중합)
        cost = self.omega_t * (t_total / self.T_max) + \
               self.omega_e * ((e_total + E_overhead) / E_local_base)

        # 보상(Reward) 설계: 실패 시 큰 페널티, 성공 시 비용의 음수
        if is_failed:
            reward = -50.0 - penalty
        else:
            reward = -cost  # 비용을 최소화해야 하므로 음수 부여

        # 한 번의 결정으로 에피소드가 끝나는 1-Step MDP 구조
        terminated = True
        truncated = False
        
        info = {"cost": cost, "is_failed": is_failed}
        
        return self.state, reward, terminated, truncated, info