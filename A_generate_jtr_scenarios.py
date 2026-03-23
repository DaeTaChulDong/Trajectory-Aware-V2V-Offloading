import os
import subprocess
import sumolib
import xml.etree.ElementTree as ET

BASE_DIR = "sumo_jtr_scenarios"
SIM_DURATION = 3600

def generate_jtr_scenarios():
    os.makedirs(BASE_DIR, exist_ok=True)
    
    # 1. 네트워크 로드
    net_file = os.path.join("sumo_scenarios", "grid.net.xml")
    if not os.path.exists(net_file):
        print(f"에러: {net_file}이 없습니다. 경로를 확인하세요.")
        return
    net = sumolib.net.readNet(net_file)
    
    # 2. 좌표 기반 외곽(Fringe) 진입 엣지 탐색 
    bbox = net.getBBoxXY()
    min_x, min_y, max_x, max_y = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
    
    fringe_edges = []
    for edge in net.getEdges():
        if edge.getFunction() == 'internal': continue
        
        from_node = edge.getFromNode()
        to_node = edge.getToNode()
        fx, fy = from_node.getCoord()
        tx, ty = to_node.getCoord()
        
        # 외곽 경계선(fringe)에 시작점이 있고, 안쪽으로 향하는 엣지 판별
        is_at_boundary = (fx <= min_x + 1 or fx >= max_x - 1 or fy <= min_y + 1 or fy >= max_y - 1)
        is_moving_inward = not (tx <= min_x + 1 or tx >= max_x - 1 or ty <= min_y + 1 or ty >= max_y - 1)
        
        if is_at_boundary and is_moving_inward:
            fringe_edges.append(edge.getID())
            
    print(f"✅ 발견된 외곽 진입 엣지: {len(fringe_edges)}개")
    if len(fringe_edges) == 0:
        print("여전히 0개입니다! 네트워크 좌표를 출력 시작")
        print(f"BBox: min({min_x}, {min_y}), max({max_x}, {max_y})")
        return

    # 3. Flow 파일 생성
    flow_file = os.path.join(BASE_DIR, "flows.xml")
    total_veh = 1000 # 1000대로 증량
    veh_per_edge = max(1, total_veh // len(fringe_edges))
    
    with open(flow_file, "w") as f:
        f.write('<routes>\n')
        for i, edge in enumerate(fringe_edges):
            f.write(f'  <flow id="f{i}" begin="0" end="{SIM_DURATION}" '
                    f'number="{veh_per_edge}" from="{edge}" '
                    f'departSpeed="max" departLane="best"/>\n')
        f.write('</routes>\n')
    
    # 4. 회전 비율별 시나리오 생성
    turn_ratios = [0.10, 0.25, 0.40, 0.55, 0.70]
    for tr in turn_ratios:
        name = f"turn_{int(tr*100):02d}"
        route_file = os.path.join(BASE_DIR, f"{name}_routes.xml")
        
        s = int((1.0 - tr) * 100)
        l = int(tr * 50)
        r = int(tr * 50)
        
        # JTRRouter 필수 규칙: 우회전, 직진, 좌회전 순서
        turn_defaults = f"{r},{s},{l}"
        
        cmd = [
            "jtrrouter",
            "-n", net_file,
            "-r", flow_file,
            "-o", route_file,
            "--turn-defaults", turn_defaults,
            "--accept-all-destinations", "true",
            "--allow-loops", "false",
            "--ignore-errors", "true",
            "--no-warnings", "true"
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 5. sumocfg 생성
        cfg_file = os.path.join(BASE_DIR, f"{name}.sumocfg")
        with open(cfg_file, "w") as f:
            f.write(f"""<configuration>
    <input>
        <net-file value="../sumo_scenarios/grid.net.xml"/>
        <route-files value="{name}_routes.xml"/>
    </input>
    <time><begin value="0"/><end value="{SIM_DURATION}"/></time>
</configuration>""")
        
        # 실제 생성 차량 수 확인
        tree = ET.parse(route_file)
        v_count = len(tree.findall('.//vehicle'))
        print(f" {name} (S:{s}% L:{l}% R:{r}%) → {v_count}대 생성 완료")

if __name__ == "__main__":
    generate_jtr_scenarios()