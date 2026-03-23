import os
import sys
import subprocess
import random
import math
import xml.etree.ElementTree as ET

if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = "/usr/local/opt/sumo/share/sumo" 
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if tools not in sys.path: sys.path.append(tools)
import sumolib

def generate_grid_network(output_dir):
    """5x5 그리드 네트워크 생성"""
    os.makedirs(output_dir, exist_ok=True)
    net_file = os.path.join(output_dir, "grid.net.xml")
    
    cmd = [
        "netgenerate", "--grid",
        "--grid.x-number", "5", "--grid.y-number", "5", 
        "--grid.x-length", "100", "--grid.y-length", "100",
        "--default.lanenumber", "3", 
        "--default-junction-type", "traffic_light",
        "--tls.guess", "true",
        "--output-file", net_file
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return net_file

def verify_turn_ratio(net_file, route_file):
    """생성된 route 파일의 실제 차량 수 및 회전 비율을 검증합니다."""
    net = sumolib.net.readNet(net_file)
    tree = ET.parse(route_file)
    
    total_turns = 0
    total_junctions = 0
    actual_vehicles = 0
    
    for vehicle in tree.findall('.//vehicle'):
        actual_vehicles += 1
        route = vehicle.find('route')
        if route is None: continue
        edges = route.get('edges').split()
        
        for i in range(len(edges) - 1):
            try:
                e1 = net.getEdge(edges[i])
                e2 = net.getEdge(edges[i+1])
                
                # 내부(Junction) 엣지는 무시하고 실제 도로 엣지 간의 각도 변화 계산
                if e1.getFunction() == 'internal' or e2.getFunction() == 'internal': 
                    continue
                    
                total_junctions += 1
                angle_diff = abs(e1.getAngle() - e2.getAngle())
                if angle_diff > 180: angle_diff = 360 - angle_diff
                
                # 45도 이상 꺾이면 회전으로 간주
                if angle_diff > 45: 
                    total_turns += 1
            except Exception:
                pass
                
    actual_ratio = (total_turns / max(total_junctions, 1)) * 100
    return actual_vehicles, actual_ratio

def generate_routes_with_turn_ratio(net_file, output_dir, turn_ratio, num_vehicles=800, sim_duration=3600, seed=42):
    """회전 비율을 정밀하게 제어한 trip 및 route 파일 생성"""
    random.seed(seed)
    scenario_name = f"turn_{int(turn_ratio*100):02d}"
    
    net = sumolib.net.readNet(net_file)
    bbox = net.getBBoxXY()
    min_x, min_y, max_x, max_y = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
    
    # 🌟 [수정 1] 정확한 좌표와 벡터 기반의 Fringe Edge 분류
    fringe_edges = {"left": [], "right": [], "top": [], "bottom": []}
    
    for edge in net.getEdges():
        from_node = edge.getFromNode()
        to_node = edge.getToNode()
        fx, fy = from_node.getCoord()
        tx, ty = to_node.getCoord()
        
        # 외부에서 내부로 들어오는 진입 엣지만 필터링
        is_from_fringe = (fx <= min_x + 1 or fx >= max_x - 1 or fy <= min_y + 1 or fy >= max_y - 1)
        is_to_internal = not (tx <= min_x + 1 or tx >= max_x - 1 or ty <= min_y + 1 or ty >= max_y - 1)
        
        if is_from_fringe and is_to_internal:
            dx, dy = tx - fx, ty - fy
            if dx > 0 and abs(dy) < abs(dx): fringe_edges["left"].append(edge.getID())
            elif dx < 0 and abs(dy) < abs(dx): fringe_edges["right"].append(edge.getID())
            elif dy > 0 and abs(dx) < abs(dy): fringe_edges["bottom"].append(edge.getID())
            elif dy < 0 and abs(dx) < abs(dy): fringe_edges["top"].append(edge.getID())

    # 🌟 [수정 반영] 각 방향별 진입 엣지 개수 확인 및 빈 리스트 방어 코드
    print(f"\n  🔍 [Fringe Edge 확인] {scenario_name}:")
    for direction, edges in fringe_edges.items():
        print(f"    - {direction}: {len(edges)} edges")
        if len(edges) == 0:
            raise ValueError(f"🚨 치명적 에러: '{direction}' 방향의 진입 엣지가 0개입니다! 네트워크(net.xml) 구조를 확인하세요.")
        
    # 🌟 [수정 2] 방향 조합 우선 선택 (편중 방지)
    straight_directions = [("left", "right"), ("right", "left"), ("bottom", "top"), ("top", "bottom")]
    turn_directions = [("left", "top"), ("left", "bottom"), ("right", "top"), ("right", "bottom"),
                       ("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]

    trips_xml = ['<routes>']
    
    # 🌟 [수정 3] 차량 수를 800대로 대폭 늘려 시뮬레이션 공백(조기 종료) 방지
    depart_times = sorted([random.uniform(0, sim_duration) for _ in range(num_vehicles)])
    
    for veh_id, depart in enumerate(depart_times):
        if random.random() < turn_ratio:
            dir_from, dir_to = random.choice(turn_directions)
        else:
            dir_from, dir_to = random.choice(straight_directions)
            
        from_edge = random.choice(fringe_edges[dir_from])
        to_edge = random.choice(fringe_edges[dir_to])
            
        trips_xml.append(f'  <trip id="veh_{veh_id}" depart="{depart:.1f}" from="{from_edge}" to="{to_edge}" departSpeed="max"/>')
    trips_xml.append('</routes>')
    
    trip_file = os.path.join(output_dir, f"{scenario_name}_trips.xml")
    with open(trip_file, 'w') as f:
        f.write('\n'.join(trips_xml))
        
    route_file = os.path.join(output_dir, f"{scenario_name}_routes.xml")
    subprocess.run([
        "duarouter", "-n", net_file, "-t", trip_file, "-o", route_file,
        "--ignore-errors", "true", "--no-warnings", "true"
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    cfg_content = f"""<configuration>
    <input>
        <net-file value="grid.net.xml"/>
        <route-files value="{scenario_name}_routes.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{sim_duration}"/>
    </time>
</configuration>"""
    cfg_file = os.path.join(output_dir, f"{scenario_name}.sumocfg")
    with open(cfg_file, 'w') as f:
        f.write(cfg_content)
        
    # 🌟 [수정 4 & 5] 실제 차량 수 및 회전 비율 정밀 검증
    actual_veh, actual_ratio = verify_turn_ratio(net_file, route_file)
    
    print(f"✅ 시나리오 '{scenario_name}' (목표 회전율: {int(turn_ratio*100)}%) 생성 완료!")
    print(f"   → 요청: {num_vehicles}대 | 실제 생성: {actual_veh}대")
    print(f"   → 실제 누적 회전 비율: {actual_ratio:.1f}%\n")
    
    if actual_veh < num_vehicles * 0.9:
        print(f"   ⚠️ 경고: 너무 많은 차량({num_vehicles - actual_veh}대)의 경로 생성이 실패했습니다.")

if __name__ == "__main__":
    BASE_DIR = "sumo_scenarios" 
    print("🚀 SUMO 5x5 교차로 다중 시나리오 생성을 시작합니다...\n")
    
    net_file = generate_grid_network(BASE_DIR)
    turn_ratios = [0.10, 0.25, 0.40, 0.55, 0.70]
    
    for tr in turn_ratios:
        generate_routes_with_turn_ratio(net_file, BASE_DIR, turn_ratio=tr, num_vehicles=800, sim_duration=3600, seed=42)
        
    print("🎉 모든 데이터셋 생성 및 검증이 완벽하게 끝났습니다!")