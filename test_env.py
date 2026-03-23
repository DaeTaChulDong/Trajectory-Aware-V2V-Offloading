from envs.v2v_env import V2VOffloadingEnv

# 1. 환경 생성
env = V2VOffloadingEnv(num_svs=10)

# 2. 환경 초기화
state, info = env.reset()
print(f"초기 State 형태: {state.shape}")
print(f"Action Mask (안전한 후보들): {info['action_mask']}")

# 3. 임의의 행동(Random Action) 취해보기
random_action = env.action_space.sample()
next_state, reward, terminated, truncated, step_info = env.step(random_action)

print(f"선택한 Action: {random_action}")
print(f"받은 보상(Reward): {reward:.3f}")
print(f"실패 여부: {step_info['is_failed']}")
print(f"총 비용(Cost): {step_info['cost']:.3f}")