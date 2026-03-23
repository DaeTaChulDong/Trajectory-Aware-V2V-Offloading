import os
import sys
import traci # SUMO 연동 핵심 라이브러리

# SUMO 홈 디렉토리 환경 변수 설정 (Mac 기본 경로)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("환경 변수 'SUMO_HOME'을 설정해주세요. (보통 /opt/homebrew/opt/sumo/share/sumo)")

def run_sumo_simulation():
    # 1. SUMO 실행 명령어 세팅 (GUI 모드로 실행하여 눈으로 확인)
    sumo_cmd = ["sumo-gui", "-c", "sumo_data/sim.sumocfg"]
    
    # 2. 파이썬과 SUMO 연결 시작!
    traci.start(sumo_cmd)
    
    print("🚗 SUMO 시뮬레이션 연결 성공! 데이터 추출을 시작합니다.")
    
    step = 0
    while step < 1000: # 1000 스텝(초) 동안 시뮬레이션 진행
        traci.simulationStep() # 1 프레임 앞으로 감기
        
        # 3. 현재 맵에 있는 모든 차량의 ID 가져오기
        veh_ids = traci.vehicle.getIDList()
        
        for vid in veh_ids:
            # 🌟 [논문 핵심] 차량의 실시간 정보 추출
            position = traci.vehicle.getPosition(vid) # (x, y) 좌표 -> I_zone 계산 및 통신 반경 계산용
            speed = traci.vehicle.getSpeed(vid)       # 현재 속도 (m/s)
            route = traci.vehicle.getRoute(vid)       # 예정된 주행 경로(Edge 목록) -> 궤적 임베딩용!
            
            # 예시 출력 (너무 많으면 터미널이 꽉 차니 100스텝마다 1번 차량만 출력)
            if step % 100 == 0 and vid == veh_ids[0]:
                print(f"[Step {step}] 차량 {vid} | 위치: {position} | 예정 경로: {route}")
                
        step += 1
        
    # 4. 시뮬레이션 종료
    traci.close()
    print("🏁 시뮬레이션이 무사히 종료되었습니다.")

if __name__ == "__main__":
    run_sumo_simulation()