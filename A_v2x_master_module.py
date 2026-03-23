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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = "/usr/local/opt/sumo/share/sumo" 
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if tools not in sys.path: sys.path.append(tools)
import sumolib
import traci

class V2XConfig:
    def __init__(self):
        self.use_masking = True
        self.use_embedding = True  
        self.task_scale = 1.0
        self.max_svs = 10
        self.alpha = 0.4  
        self.beta = 0.6   
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
    
    traci.start(["sumo", "-c", sumocfg_path, "--seed", str(seed), "--no-warnings", "--no-step-log", "--quit-on-end"])
    
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
            for (v1, v2), (start_t, r1, r2) in active_pairs.items():
                both_alive = (v1 in pos_dict and v2 in pos_dict)
                if not both_alive:
                    ended_pairs.append(((v1, v2), False)) 
                else:
                    dist = math.hypot(pos_dict[v1][0]-pos_dict[v2][0], pos_dict[v1][1]-pos_dict[v2][1])
                    if dist > 100.0:
                        ended_pairs.append(((v1, v2), True)) 
            
            for pair, is_valid in ended_pairs:
                start_t, r1, r2 = active_pairs.pop(pair)
                if is_valid:
                    actual_conn_time = current_time - start_t
                    if actual_conn_time > 1.0: 
                        target_t_norm = min(actual_conn_time, 30.0) / 30.0
                        dataset.append((r1, r2, target_t_norm))
            
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
                                active_pairs[(v1, v2)] = (current_time, r1, r2)
    finally:
        traci.close()
        
    print(f"[Phase 1-A] 완료. 총 {len(dataset)} 쌍의 실제 궤적 데이터 수집됨.")
    
    confs = [d[2] for d in dataset]
    if len(confs) > 0:
        print(f"  [데이터 분포 진단] Normalized Target Time (0~1):")
        print(f"     - Mean: {np.mean(confs):.3f}")
        print(f"     - Median: {np.median(confs):.3f}")
        print(f"     - < 0.5 비율 (<15초 단절): {np.mean(np.array(confs) < 0.5) * 100:.1f}%")
        print(f"     - == 1.0 비율 (30초 이상 생존): {np.mean(np.array(confs) >= 0.99) * 100:.1f}%")
    
    print("\n[Phase 1-B] 수집된 데이터로 인코더 및 Predictor 훈련 시작...")
    encoder = IntentionEncoder()
    predictor = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid()) 
    optimizer = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=0.001)
    
    epochs = 20
    for epoch in range(epochs):
        total_loss = 0
        np.random.shuffle(dataset)
        batch = dataset[:2000] 
        
        for r1, r2, target_t_norm in batch:
            optimizer.zero_grad()
            emb1, emb2 = encoder(r1), encoder(r2)
            sim = F.cosine_similarity(emb1, emb2)
            
            predicted_norm = predictor(sim.unsqueeze(-1))
            target_t = torch.tensor([float(target_t_norm)], dtype=torch.float32)
            
            loss = F.mse_loss(predicted_norm.squeeze(), target_t.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  - Epoch [{epoch+1:02d}/{epochs}] Avg MSE Loss (Normalized Time): {total_loss / len(batch):.4f}")
            
    if len(dataset) >= 500:
        with torch.no_grad():
            errors = []
            for r1, r2, target_t_norm in dataset[-500:]:
                emb1, emb2 = encoder(r1), encoder(r2)
                sim = F.cosine_similarity(emb1, emb2)
                pred_norm = predictor(sim.unsqueeze(-1)).item()
                errors.append(abs(pred_norm - target_t_norm))
            print(f"\n  [검증] Predictor MAE: {np.mean(errors):.4f} (Normalized Time 0~1 스케일)")
            
    for param in encoder.parameters(): param.requires_grad = False
    for param in predictor.parameters(): param.requires_grad = False
    print("[Phase 1-B] 사전 학습 완료. 인코더 및 Predictor 가중치 고정됨.\n")
    return encoder, predictor

# A_v2x_master_module.py 파일 내 SumoV2XEnv 클래스 전체 덮어쓰기

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
        binary = "sumo-gui" if gui else "sumo"
        sumo_cmd = [binary, "-c", self.sumocfg_path, "--seed", str(self.sim_seed), "--no-warnings", "--no-step-log", "--quit-on-end"]
        traci.start(sumo_cmd)
        
    def close_sumo(self):
        traci.close()

    def _estimate_actual_connection(self, sv_vid, physical_max):
        try:
            sv_route = traci.vehicle.getRoute(sv_vid)
            sv_idx = traci.vehicle.getRouteIndex(sv_vid)
            tv_route = traci.vehicle.getRoute(self.tv_id)
            tv_idx = traci.vehicle.getRouteIndex(self.tv_id)
            
            # 현재 엣지 이후의 향후 경로만 비교 (현재 엣지는 제외)
            sv_future = sv_route[sv_idx+1 : sv_idx+4]
            tv_future = tv_route[tv_idx+1 : tv_idx+4]
            
            if not sv_future or not tv_future:
                return physical_max
            
            # 향후 경로의 겹침 비율로 GT 산출
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
                
                # 1. 환경 채점용 Ground Truth
                self.T_conn_gt[sv_count] = self._estimate_actual_connection(vid, max_time)
                
                # 2. 마스킹용 Predicted 산출: Binary Risk Detector 적용
                if self.config.use_embedding and self.encoder and self.predictor and tv_emb is not None:
                    sv_emb = self.encoder(sv_raw)
                    similarity = F.cosine_similarity(tv_emb, sv_emb)
                    predicted_norm = self.predictor(similarity.unsqueeze(-1)).item()
                    
                    # 사전학습 데이터 분포의 중앙값(Median) 부근을 참고하여 임계값 설정
                    RISK_THRESHOLD = 0.40  
                    
                    if predicted_norm < RISK_THRESHOLD:
                        # [위험 감지] 이 차는 교차로에서 찢어질 확률이 높음. 보수적 필터링 적용
                        self.T_conn_predicted[sv_count] = min(max_time * 0.3, 30.0)
                    else:
                        # [안전 판정] 이 차는 같이 갈 확률이 높음. Physical과 동일하게 낙관적 허용
                        self.T_conn_predicted[sv_count] = min(max_time, 30.0)
                else:
                    # Physical은 모든 차량을 낙관적으로 평가
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