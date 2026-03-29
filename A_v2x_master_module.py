import os
import sys
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt 
import subprocess
import socket
import time

SUMO_BINARY = "/opt/anaconda3/envs/v2v_rl/lib/python3.10/site-packages/sumo/bin/sumo"
os.environ['SUMO_HOME'] = "/opt/anaconda3/envs/v2v_rl/lib/python3.10/site-packages/sumo/share/sumo"
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = 'SUMO_HOME'
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if tools not in sys.path: sys.path.append(tools)

import sumolib
import traci

def start_sumo_traci(sumocfg_path, sumo_binary=SUMO_BINARY):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen(
        [sumo_binary, "-c", sumocfg_path,
         "--remote-port", str(port),
         "--no-warnings", "--no-step-log", "--quit-on-end"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    for _ in range(30):
        try:
            traci.init(port=port)
            return proc
        except:
            time.sleep(0.5)
    raise RuntimeError("SUMO TraCI 연결 실패")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class V2XConfig:
    def __init__(self):
        self.use_masking = True
        self.use_embedding = True  
        self.task_scale = 1.0
        self.max_svs = 10
        self.alpha = 0.6  
        self.beta = 0.4   
        self.f_tv = 2.5
        self.f_sv_min, self.f_sv_max = 10.0, 50.0
        self.comm_range = 100.0  
        self.p_tx_dbm = 23.0       
        self.noise_dbm = -111.0    
        self.path_loss_exp = 2.7   
        self.pl_ref = 63.3         
        self.B_channel = 20e6      
        self.p_tx = 10 ** ((self.p_tx_dbm - 30) / 10.0) 
        self.kappa = 0.02          
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_ratio = 0.2
        self.ppo_epochs = 4
        self.actor_lr = 3e-4
        self.critic_lr = 1e-3
        self.max_grad_norm = 0.5   
        self.task_types = [
            {'name': 'Type 1', 'prob': 0.3, 'D': 50.0, 'C': 10.0, 'T_max': 10.0},
            {'name': 'Type 2', 'prob': 0.1, 'D': 200.0, 'C': 200.0, 'T_max': 15.0},
            {'name': 'Type 3', 'prob': 0.1, 'D': 10.0, 'C': 5.0, 'T_max': 5.0},
            {'name': 'Type 4', 'prob': 0.3, 'D': 100.0, 'C': 500.0, 'T_max': 20.0},
            {'name': 'Type 5', 'prob': 0.2, 'D': 500.0, 'C': 100.0, 'T_max': 10.0}
        ]

class RolloutBuffer:
    def __init__(self):
        self.states, self.actions, self.logprobs = [], [], []
        self.rewards, self.values, self.masks = [], [], []
        self.dones = []
    def clear(self):
        del self.states[:], self.actions[:], self.logprobs[:]
        del self.rewards[:], self.values[:], self.masks[:]
        del self.dones[:]

class IntentionEncoder(nn.Module):
    def __init__(self, input_dim=6, latent_dim=16):
        super(IntentionEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
    def forward(self, raw_data):
        latent = self.encoder(raw_data)
        return F.normalize(latent, p=2, dim=1)

class ConnectionPredictor(nn.Module):
    def __init__(self):
        super(ConnectionPredictor, self).__init__()
        # 입력 차원 4: [cosine_similarity, dist_norm, rel_speed_norm, heading_diff_norm]
        self.net = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )
    
    def forward(self, similarity, phys_info):
        # similarity는 1D 텐서이므로 차원을 늘려 결합
        x = torch.cat([similarity.unsqueeze(-1), phys_info], dim=-1)
        return self.net(x)

def _get_raw_features(vid, net):
    pos = traci.vehicle.getPosition(vid)
    speed = traci.vehicle.getSpeed(vid)
    heading = traci.vehicle.getAngle(vid) / 360.0
    route = traci.vehicle.getRoute(vid)
    dest_edge_id = route[-1]
    
    try:
        edge = net.getEdge(dest_edge_id)
        dest_x, dest_y = edge.getToNode().getCoord()
    except:
        dest_x, dest_y = pos[0], pos[1] 
        
    return torch.FloatTensor([[pos[0]/500.0, pos[1]/500.0, speed/30.0, heading, dest_x/500.0, dest_y/500.0]])

def pretrain_intention_encoder(sumocfg_path, num_steps=5000, seed=42):
    print("\n[Phase 1-A] SUMO 시뮬레이션에서 실제 연결 지속 시간(Actual T_conn) 데이터 수집 중...")
    
    tree = ET.parse(sumocfg_path)
    net_file_rel = tree.find('.//net-file').get('value')
    net_file_abs = os.path.join(os.path.dirname(sumocfg_path), net_file_rel)
    net = sumolib.net.readNet(net_file_abs)
    
    sumo_proc = start_sumo_traci(sumocfg_path)

    active_pairs = {} 
    dataset = []      
    
    try:
        for step in range(num_steps):
            traci.simulationStep()
            if traci.simulation.getMinExpectedNumber() <= 0: break
                
            veh_ids = traci.vehicle.getIDList()
            current_time = traci.simulation.getTime()
            pos_dict = {vid: traci.vehicle.getPosition(vid) for vid in veh_ids}
            
            ended_pairs = []
            for (v1, v2), (start_t, r1, r2, phys_info) in active_pairs.items():
                both_alive = (v1 in pos_dict and v2 in pos_dict)
                if not both_alive:
                    ended_pairs.append(((v1, v2), False)) 
                else:
                    dist = math.hypot(pos_dict[v1][0]-pos_dict[v2][0], pos_dict[v1][1]-pos_dict[v2][1])
                    if dist > 100.0:
                        ended_pairs.append(((v1, v2), True)) 
            
            for pair, is_valid in ended_pairs:
                start_t, r1, r2, phys_info = active_pairs.pop(pair)
                if is_valid:
                    actual_conn_time = current_time - start_t
                    if actual_conn_time > 1.0: 
                        target_t_norm = min(actual_conn_time, 30.0) / 30.0
                        dataset.append((r1, r2, phys_info, target_t_norm))
            
            if len(veh_ids) >= 2:
                sample_size = min(len(veh_ids), 20)
                sampled_vids = np.random.choice(veh_ids, sample_size, replace=False)
                for i in range(len(sampled_vids)):
                    for j in range(i+1, len(sampled_vids)):
                        v1, v2 = sampled_vids[i], sampled_vids[j]
                        if (v1, v2) not in active_pairs and (v2, v1) not in active_pairs:
                            dist = math.hypot(pos_dict[v1][0]-pos_dict[v2][0], pos_dict[v1][1]-pos_dict[v2][1])
                            if dist <= 100.0:
                                r1 = _get_raw_features(v1, net)
                                r2 = _get_raw_features(v2, net)
                                
                                spd1, spd2 = traci.vehicle.getSpeed(v1), traci.vehicle.getSpeed(v2)
                                rel_speed = abs(spd1 - spd2) / 30.0
                                dist_norm = dist / 100.0
                                h1, h2 = traci.vehicle.getAngle(v1) / 360.0, traci.vehicle.getAngle(v2) / 360.0
                                heading_diff = abs(h1 - h2)
                                if heading_diff > 0.5: heading_diff = 1.0 - heading_diff
                                
                                phys_info = torch.FloatTensor([[dist_norm, rel_speed, heading_diff]])
                                active_pairs[(v1, v2)] = (current_time, r1, r2, phys_info)
    finally:
        traci.close()
        
    print(f"[Phase 1-A] 완료. 총 {len(dataset)} 쌍의 실제 궤적 데이터 수집됨.")
    
    print("\n[Phase 1-B] 수집된 데이터로 인코더 및 Predictor 훈련 시작...")
    
    # 🌟 데이터 균형 맞추기 (오버샘플링)
    short_data = [d for d in dataset if d[3] < 0.33]   # 10초 미만
    mid_data = [d for d in dataset if 0.33 <= d[3] < 0.67]  # 10~20초
    long_data = [d for d in dataset if d[3] >= 0.67]    # 20초 이상
    
    print(f"  [데이터 균형] 원본 분포: 짧은={len(short_data)}, 중간={len(mid_data)}, 긴={len(long_data)}")
    
    max_group = max(len(short_data), len(mid_data), len(long_data))
    
    if len(short_data) > 0:
        short_oversampled = short_data * (max_group // len(short_data)) + \
                           short_data[:max_group % len(short_data)]
    else:
        short_oversampled = []
    
    if len(mid_data) > 0:
        mid_oversampled = mid_data * (max_group // len(mid_data)) + \
                         mid_data[:max_group % len(mid_data)]
    else:
        mid_oversampled = []
        
    balanced_dataset = short_oversampled + mid_oversampled + long_data
    print(f"  [데이터 균형] 균형 후: 짧은={len(short_oversampled)}, 중간={len(mid_oversampled)}, 긴={len(long_data)}, 총={len(balanced_dataset)}")
    
    encoder = IntentionEncoder()
    predictor = ConnectionPredictor()
    optimizer = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=0.001)
    
    # 🌟 에폭 수 증가 + 학습률 스케줄러 추가
    epochs = 50
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    for epoch in range(epochs):
        total_loss = 0
        np.random.shuffle(balanced_dataset)
        batch = balanced_dataset[:3000]
        
        for r1, r2, phys_info, target_t_norm in batch:
            optimizer.zero_grad()
            emb1, emb2 = encoder(r1), encoder(r2)
            sim = F.cosine_similarity(emb1, emb2)
            
            predicted_norm = predictor(sim, phys_info)
            target_t = torch.tensor([[float(target_t_norm)]], dtype=torch.float32)
            
            loss = F.mse_loss(predicted_norm, target_t)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  - Epoch [{epoch+1:02d}/{epochs}] Avg MSE Loss: {total_loss / len(batch):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # ==========================================
    # 🌟 [Phase 1-C] Predictor 심층 성능 검증 (신규 추가)
    # ==========================================
    if len(dataset) >= 500:
        print("\n" + "="*60)
        print(" [심층 검증] 인코더 & Predictor 단독 성능 진단")
        print("="*60)
        with torch.no_grad():
            preds, targets, sims = [], [], []
            for r1, r2, phys_info, target_t_norm in dataset[-500:]:
                emb1, emb2 = encoder(r1), encoder(r2)
                sim = F.cosine_similarity(emb1, emb2)
                pred_norm = predictor(sim, phys_info).item()
                
                preds.append(pred_norm)
                targets.append(target_t_norm)
                sims.append(sim.item())
                
            preds = np.array(preds)
            targets = np.array(targets)
            sims = np.array(sims)
            actual_times = targets * 30.0
            
            # 1. 유사도 변별력 확인
            short_conn = sims[actual_times < 10.0]
            long_conn = sims[actual_times >= 20.0]
            mean_sim_short = np.mean(short_conn) if len(short_conn) > 0 else 0
            mean_sim_long = np.mean(long_conn) if len(long_conn) > 0 else 0
            print(f" 유사도 변별력 (긴 연결 vs 짧은 연결):")
            print(f"   - 짧은 연결(<10초) 평균 유사도: {mean_sim_short:.3f}")
            print(f"   - 긴 연결(>=20초) 평균 유사도: {mean_sim_long:.3f}")
            print(f"   - 유사도 차이: {mean_sim_long - mean_sim_short:.3f} (목표: > 0.2)")

            # 2. Binary Risk Detector 정확도 (Threshold = 0.4)
            target_binary = (targets < 0.4).astype(int)
            pred_binary = (preds < 0.4).astype(int)
            accuracy = np.mean(target_binary == pred_binary) * 100
            print(f"\n Binary 위험 감지 정확도 (Threshold=0.4): {accuracy:.1f}% (목표: > 70%)")
            
            # 3. 구간별 평균 절대 오차(MAE)
            errs = np.abs(preds - targets) * 30.0 
            idx_0_5 = (actual_times >= 0) & (actual_times < 5)
            idx_5_15 = (actual_times >= 5) & (actual_times < 15)
            idx_15_30 = (actual_times >= 15) & (actual_times <= 30)
            
            print(f"\n 구간별 평균 절대 오차 (MAE, 초 단위):")
            print(f"   - [ 0~ 5초] 구간 오차: {np.mean(errs[idx_0_5]):.2f}초" if np.sum(idx_0_5) > 0 else "   - [0~5초] 데이터 없음")
            print(f"   - [ 5~15초] 구간 오차: {np.mean(errs[idx_5_15]):.2f}초" if np.sum(idx_5_15) > 0 else "   - [5~15초] 데이터 없음")
            print(f"   - [15~30초] 구간 오차: {np.mean(errs[idx_15_30]):.2f}초" if np.sum(idx_15_30) > 0 else "   - [15~30초] 데이터 없음")

            # 4. 산점도(Scatter Plot) 시각화 및 저장
            plt.figure(figsize=(7, 6))
            plt.scatter(targets * 30.0, preds * 30.0, alpha=0.5, color='royalblue', edgecolors='w', s=60)
            plt.plot([0, 30], [0, 30], 'r--', lw=2, label='Perfect Prediction (y=x)')
            plt.title('Predictor Performance: Actual vs. Predicted $T_{conn}$', fontsize=14, fontweight='bold')
            plt.xlabel('Actual Connection Time (sec)', fontsize=12)
            plt.ylabel('Predicted Connection Time (sec)', fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig('predictor_validation_scatter.png', dpi=300)
            print(f"\n 산점도 그래프 저장 완료: 'predictor_validation_scatter.png'")
            print("="*60 + "\n")
            
    for param in encoder.parameters(): param.requires_grad = False
    for param in predictor.parameters(): param.requires_grad = False
    print("[Phase 1-B] 사전 학습 및 검증 완료. 모델 가중치 고정됨.\n")
    return encoder, predictor


class SumoV2XEnv:
    def __init__(self, sumocfg_path, config: V2XConfig, pretrained_encoder=None, pretrained_predictor=None, sumo_seed=42):
        self.sumocfg_path = sumocfg_path
        self.config = config
        self.state_dim = 2 + (self.config.max_svs * 3) 
        self.action_dim = self.config.max_svs + 1      
        self.encoder = pretrained_encoder
        self.predictor = pretrained_predictor
        self.sim_seed = sumo_seed 
        
        self.reload_count = 0
        
        tree = ET.parse(sumocfg_path)
        net_file_rel = tree.find('.//net-file').get('value')
        self.net_file_abs = os.path.join(os.path.dirname(sumocfg_path), net_file_rel)
        self.net = sumolib.net.readNet(self.net_file_abs)
        
        self.last_action_mask = np.zeros(self.action_dim)
        self.last_num_svs = 0

        self.T_conn_gt = np.zeros(self.config.max_svs)
        self.T_conn_predicted = np.zeros(self.config.max_svs)

    def start_sumo(self, gui=False):
        self.sumo_proc = start_sumo_traci(self.sumocfg_path)
        binary = SUMO_BINARY
        sumo_cmd = [binary, "-c", self.sumocfg_path, "--seed", str(self.sim_seed), 
                "--no-warnings", "--no-step-log", "--quit-on-end"]
        traci.start(sumo_cmd)
        
    def close_sumo(self):
        traci.close()

    def _estimate_actual_connection(self, sv_vid, physical_max):
        try:
            sv_route = traci.vehicle.getRoute(sv_vid)
            sv_idx = traci.vehicle.getRouteIndex(sv_vid)
            tv_route = traci.vehicle.getRoute(self.tv_id)
            tv_idx = traci.vehicle.getRouteIndex(self.tv_id)
            
            sv_future = sv_route[sv_idx+1 : sv_idx+4]
            tv_future = tv_route[tv_idx+1 : tv_idx+4]
            
            if not sv_future or not tv_future:
                return physical_max
            
            common = len(set(sv_future) & set(tv_future))
            total = max(len(sv_future), len(tv_future))
            ratio = common / max(total, 1) 
            
            return physical_max * max(0.3, ratio)
        except:
            return physical_max

    def reset(self):
        veh_ids = []
        while len(veh_ids) < 2:
            traci.simulationStep()
            if traci.simulation.getMinExpectedNumber() <= 0:
                self.reload_count += 1
                new_seed = self.sim_seed + (self.reload_count * 1000)
                traci.load(["-c", self.sumocfg_path, "--seed", str(new_seed), "--no-warnings", "--no-step-log"])
            veh_ids = traci.vehicle.getIDList()
            
        self.tv_id = np.random.choice(veh_ids)
        tv_raw = _get_raw_features(self.tv_id, self.net)
        
        tv_emb = None
        if self.encoder: tv_emb = self.encoder(tv_raw)
        
        t_idx = np.random.choice(len(self.config.task_types), p=[t['prob'] for t in self.config.task_types])
        self.task = dict(self.config.task_types[t_idx])
        self.task['D'] *= self.config.task_scale
        self.task['C'] *= self.config.task_scale
        
        self.sv_list, self.f_sv, self.R_sv = [], np.zeros(self.config.max_svs), np.zeros(self.config.max_svs)
        self.T_conn_gt.fill(0)
        self.T_conn_predicted.fill(0)
        
        sv_count = 0
        for vid in veh_ids:
            if vid == self.tv_id or sv_count >= self.config.max_svs: continue
            
            sv_raw = _get_raw_features(vid, self.net)
            pos1, pos2 = traci.vehicle.getPosition(self.tv_id), traci.vehicle.getPosition(vid)
            dist = max(5.0, math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])) 
            
            if dist <= self.config.comm_range:
                self.sv_list.append(vid)
                self.f_sv[sv_count] = np.random.uniform(self.config.f_sv_min, self.config.f_sv_max)
                
                pl_db = self.config.pl_ref + 10.0 * self.config.path_loss_exp * math.log10(dist)
                fading_linear = max(1e-4, np.random.rayleigh(1.0)**2)
                fading_db = 10.0 * math.log10(fading_linear)
                
                snr_db = self.config.p_tx_dbm - pl_db + fading_db - self.config.noise_dbm
                snr_linear = 10.0 ** (snr_db / 10.0)
                
                r_mbps = (self.config.B_channel / 1e6) * math.log2(1.0 + snr_linear)
                self.R_sv[sv_count] = min(max(1.0, r_mbps), 50.0) 
                
                rel_speed = abs(tv_raw[0][2].item() - sv_raw[0][2].item()) * 30.0
                rel_speed_safe = max(rel_speed, 0.1)
                max_time = min((100.0 - dist) / rel_speed_safe, 30.0)
                
                self.T_conn_gt[sv_count] = self._estimate_actual_connection(vid, max_time)
                
                if self.config.use_embedding and self.encoder and self.predictor and tv_emb is not None:
                    sv_emb = self.encoder(sv_raw)
                    similarity = F.cosine_similarity(tv_emb, sv_emb)
                    
                    spd_tv, spd_sv = tv_raw[0][2].item() * 30.0, sv_raw[0][2].item() * 30.0
                    rel_speed_dyn = abs(spd_tv - spd_sv) / 30.0
                    dist_norm = dist / 100.0
                    h_tv, h_sv = tv_raw[0][3].item(), sv_raw[0][3].item()
                    heading_diff = abs(h_tv - h_sv)
                    if heading_diff > 0.5: heading_diff = 1.0 - heading_diff
                    
                    phys_info = torch.FloatTensor([[dist_norm, rel_speed_dyn, heading_diff]])
                    
                    # 버그 수정 완료: unsqueeze(-1) 제거! (내부에서 차원 확장됨)
                    predicted_norm = self.predictor(similarity, phys_info).item()
                    
                    RISK_THRESHOLD = 0.40  
                    if predicted_norm < RISK_THRESHOLD:
                        self.T_conn_predicted[sv_count] = min(max_time * 0.3, 30.0)
                    else:
                        self.T_conn_predicted[sv_count] = min(max_time, 30.0)
                else:
                    self.T_conn_predicted[sv_count] = min(max_time, 30.0)
                
                sv_count += 1
                
        state = np.zeros(self.state_dim, dtype=np.float32)
        state[0], state[1] = self.task['D'] / 100.0, self.task['C'] / 100.0
        state[2:2+self.config.max_svs] = self.f_sv / 50.0
        state[2+self.config.max_svs:2+2*self.config.max_svs] = self.R_sv / 100.0
        state[2+2*self.config.max_svs:] = self.T_conn_predicted / 30.0
        
        action_mask = np.ones(self.action_dim, dtype=np.float32) 
        if self.config.use_masking:
            action_mask = np.zeros(self.action_dim, dtype=np.float32)
            if (self.task['C'] / self.config.f_tv) <= self.task['T_max']: action_mask[0] = 1.0
            for j in range(sv_count):
                if (self.task['D'] / self.R_sv[j] + self.task['C'] / self.f_sv[j]) <= min(self.task['T_max'], self.T_conn_predicted[j]):
                    action_mask[j+1] = 1.0
            if np.sum(action_mask) == 0: action_mask[0] = 1.0
        
        self.last_action_mask = action_mask
        self.last_num_svs = sv_count
                
        return state, action_mask

    def step(self, action):
        D_i, C_i, T_max = self.task['D'], self.task['C'], self.task['T_max']
        is_failed = False
        t_trans, t_comp, e_trans, e_comp = 0.0, 0.0, 0.0, 0.0
        
        if action == 0:
            t_comp = C_i / self.config.f_tv
            e_comp = self.config.kappa * C_i * (self.config.f_tv ** 2) 
            if t_comp > T_max: is_failed = True
        else:
            sv_idx = action - 1
            if sv_idx >= len(self.sv_list): is_failed = True
            else:
                t_trans = D_i / self.R_sv[sv_idx]
                t_comp = C_i / self.f_sv[sv_idx]
                e_trans = self.config.p_tx * t_trans
                total_time = t_trans + t_comp
                
                if total_time > T_max or total_time > self.T_conn_gt[sv_idx]:
                    is_failed = True
                    
        cost = self.config.alpha * (t_trans + t_comp) + self.config.beta * (e_trans + e_comp)
        final_cost = 100.0 if is_failed else cost
        reward = -final_cost
        
        done = True 
        next_state = np.zeros(self.state_dim, dtype=np.float32) 
        
        info = {
            'cost': final_cost, 
            'failed': is_failed,
            't_trans': t_trans, 
            't_comp': t_comp,
            'e_trans': e_trans, 
            'e_comp': e_comp,
            'task_type': self.task['name'],
            'num_valid_actions': int(np.sum(self.last_action_mask)),
            'num_svs': self.last_num_svs
        }
        traci.simulationStep()
        return next_state, reward, done, info

class FullPPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim, config: V2XConfig):
        super(FullPPOAgent, self).__init__()
        self.config = config
        self.actor = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, action_dim))
        self.critic = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
        
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=config.critic_lr)

    def get_action(self, state, action_mask, deterministic=False):
        logits = self.actor(state)
        logits = logits - (1.0 - action_mask) * 1e9 
        probs = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else probs.sample()
        return action, probs.log_prob(action), self.critic(state)

    def update(self, buffer: RolloutBuffer):
        states = torch.FloatTensor(np.array(buffer.states))
        actions = torch.LongTensor(np.array(buffer.actions))
        old_logprobs = torch.stack(buffer.logprobs).detach()
        rewards = buffer.rewards
        dones = buffer.dones
        masks = torch.FloatTensor(np.array(buffer.masks))
        
        returns, advantages = [], []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(dones)):
            if is_terminal: discounted_reward = 0
            discounted_reward = reward + (self.config.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
            
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7) 
        
        for _ in range(self.config.ppo_epochs):
            logits = self.actor(states)
            logits = logits - (1.0 - masks) * 1e9
            probs = Categorical(logits=logits)
            logprobs = probs.log_prob(actions)
            dist_entropy = probs.entropy()
            state_values = self.critic(states).squeeze()
            
            advantages = returns - state_values.detach()
            
            ratios = torch.exp(logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * dist_entropy.mean()
            critic_loss = nn.MSELoss()(state_values, returns)
            
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            self.optimizer_actor.step()
            
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
            self.optimizer_critic.step()

class VanillaPPOAgent(FullPPOAgent):
    def get_action(self, state, action_mask, deterministic=False):
        no_mask = torch.ones_like(action_mask)
        return super().get_action(state, no_mask, deterministic)

class GreedySNRAgent:
    def __init__(self, state_dim, action_dim, config):
        pass 

    def get_action(self, state, action_mask, deterministic=True):
        state_np = state.squeeze().numpy()
        max_svs = int((len(state_np) - 2) / 3)
        R_sv_array = state_np[2+max_svs : 2+2*max_svs]
        
        mask_np = action_mask.squeeze().numpy()
        valid_actions = np.where(mask_np[1:] == 1.0)[0] 
        if len(valid_actions) == 0: return torch.tensor(0), None, None
            
        best_sv_idx = valid_actions[np.argmax(R_sv_array[valid_actions])]
        return torch.tensor(best_sv_idx + 1), None, None

class GreedyCPUAgent:
    def __init__(self, state_dim, action_dim, config):
        pass

    def get_action(self, state, action_mask, deterministic=True):
        state_np = state.squeeze().numpy()
        max_svs = int((len(state_np) - 2) / 3)
        f_sv_array = state_np[2 : 2+max_svs]
        
        mask_np = action_mask.squeeze().numpy()
        valid_actions = np.where(mask_np[1:] == 1.0)[0]
        if len(valid_actions) == 0: return torch.tensor(0), None, None
            
        best_sv_idx = valid_actions[np.argmax(f_sv_array[valid_actions])]
        return torch.tensor(best_sv_idx + 1), None, None

class GreedyStabilityAgent:
    def __init__(self, state_dim, action_dim, config):
        pass

    def get_action(self, state, action_mask, deterministic=True):
        state_np = state.squeeze().numpy()
        max_svs = int((len(state_np) - 2) / 3)
        T_conn_array = state_np[2+2*max_svs : 2+3*max_svs] 
        
        mask_np = action_mask.squeeze().numpy()
        valid_actions = np.where(mask_np[1:] == 1.0)[0]
        
        if len(valid_actions) == 0:
            return torch.tensor(0), None, None
            
        best_sv_idx = valid_actions[np.argmax(T_conn_array[valid_actions])]
        return torch.tensor(best_sv_idx + 1), None, None