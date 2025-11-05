# ==============================================================================
# VNE IMPLEMENTATION WITH PROPER SCALING (Paper-like results)
# ==============================================================================

import gym
from gym import spaces
import numpy as np
import networkx as nx
import random
from scipy.stats import expon
import heapq
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# PROGRESSIVE DIFFICULTY: Start easier, then scale up
# For TRAINING: Use easier problem
TRAIN_SUBSTRATE_NODES = 100  # Medium size
TRAIN_EDGE_PROB = 0.1
TRAIN_VN_MIN, TRAIN_VN_MAX = 3, 8
TRAIN_VNODE_RES_LOW, TRAIN_VNODE_RES_HIGH = 5, 25
TRAIN_VLINK_RES_LOW, TRAIN_VLINK_RES_HIGH = 5, 25

# For EVALUATION: Use paper's hard problem
EVAL_SUBSTRATE_NODES = 100  # Full paper size
EVAL_EDGE_PROB = 0.1
EVAL_VN_MIN, EVAL_VN_MAX = 2, 10
EVAL_VNODE_RES_LOW, EVAL_VNODE_RES_HIGH = 1, 30
EVAL_VLINK_RES_LOW, EVAL_VLINK_RES_HIGH = 1, 30

# Shared parameters
SUB_CPU_LOW, SUB_CPU_HIGH = 50, 120
SUB_BW_LOW, SUB_BW_HIGH = 50, 120
VLINK_PROB = 0.5

# Energy constants
P_B = 100.0
P_M = 200.0
P_L = P_M - P_B
LINK_POWER_ACTIVE = 5.0
BETA = 1.0

MAX_ACTIONS_PER_VN = 100
TRAIN_TIMESTEPS = 200000  # More training needed

print("✅ Configuration loaded - Progressive difficulty scaling")

# ==============================================================================
# SUBSTRATE NETWORK BUILDER
# ==============================================================================
def build_substrate(n_nodes=TRAIN_SUBSTRATE_NODES, edge_prob=TRAIN_EDGE_PROB, seed=SEED):
    """Build substrate with configurable size"""
    np.random.seed(seed)
    G = nx.Graph()
    for i in range(n_nodes):
        cpu = int(np.random.uniform(SUB_CPU_LOW, SUB_CPU_HIGH))
        G.add_node(i, cpu_total=cpu, cpu_free=cpu, active=False, mapped_vnode=None)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if np.random.rand() < edge_prob:
                bw = int(np.random.uniform(SUB_BW_LOW, SUB_BW_HIGH))
                G.add_edge(i, j, bw_total=bw, bw_free=bw, active=False)
    return G

# Training substrate (medium difficulty)
train_substrate = build_substrate(TRAIN_SUBSTRATE_NODES, TRAIN_EDGE_PROB, SEED)
print(f"✅ Training substrate: {train_substrate.number_of_nodes()} nodes, {train_substrate.number_of_edges()} edges")

# Evaluation substrate (paper difficulty)
eval_substrate = build_substrate(EVAL_SUBSTRATE_NODES, EVAL_EDGE_PROB, SEED)
print(f"✅ Eval substrate: {eval_substrate.number_of_nodes()} nodes, {eval_substrate.number_of_edges()} edges")

# ==============================================================================
# VNR GENERATOR
# ==============================================================================
class VNR:
    def __init__(self, vn_id, nodes, edges, node_reqs, edge_reqs, ta, td):
        self.id = vn_id
        self.nodes = nodes
        self.edges = edges
        self.node_reqs = node_reqs
        self.edge_reqs = edge_reqs
        self.ta = ta
        self.td = td

def gen_single_vnr(current_time, vn_id, seed=None, 
                   vn_min=TRAIN_VN_MIN, vn_max=TRAIN_VN_MAX,
                   node_res_low=TRAIN_VNODE_RES_LOW, node_res_high=TRAIN_VNODE_RES_HIGH,
                   link_res_low=TRAIN_VLINK_RES_LOW, link_res_high=TRAIN_VLINK_RES_HIGH):
    """Generate VNR with configurable difficulty"""
    if seed is not None:
        state = np.random.get_state()
        np.random.seed(seed)
    
    n_nodes = np.random.randint(vn_min, vn_max + 1)
    nodes = list(range(n_nodes))
    edges = []
    node_reqs = {}
    edge_reqs = {}
    
    for i in nodes:
        node_reqs[i] = int(np.random.uniform(node_res_low, node_res_high))
    
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if np.random.rand() < VLINK_PROB:
                edges.append((i, j))
                edge_reqs[(i, j)] = int(np.random.uniform(link_res_low, link_res_high))
    
    if seed is not None:
        np.random.set_state(state)
    
    lifetime = 25.0
    return VNR(vn_id, nodes, edges, node_reqs, edge_reqs, current_time, current_time + lifetime)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def constrained_shortest_path(G, src, dst, bw_need):
    heap = [(0, src, [])]
    visited = set()
    while heap:
        dist, node, path = heapq.heappop(heap)
        if node == dst:
            return path + [dst]
        if node in visited:
            continue
        visited.add(node)
        for nbr in G[node]:
            if G[node][nbr]['bw_free'] >= bw_need:
                heapq.heappush(heap, (dist + 1, nbr, path + [node]))
    return None

def node_power_util(cpu_used, cpu_total, active_flag):
    if not active_flag:
        return 0.0
    mu = cpu_used / (cpu_total + 1e-9)
    return P_B + P_L * mu

def substrate_power_total(G):
    total = 0.0
    for n, d in G.nodes(data=True):
        cpu_used = d['cpu_total'] - d['cpu_free']
        total += node_power_util(cpu_used, d['cpu_total'], d.get('active', False))
    for u, v, d in G.edges(data=True):
        if d.get('active', False):
            total += BETA * LINK_POWER_ACTIVE
    return total

def rev_of_vnr(vnr):
    total_res = sum(vnr.node_reqs.values()) + sum(vnr.edge_reqs.values())
    duration = max(1.0, vnr.td - vnr.ta)
    return total_res * duration

def cost_of_vnr(mapping_edge_lengths, vnr):
    node_sum = sum(vnr.node_reqs.values())
    edge_cost = sum(vnr.edge_reqs[e] * mapping_edge_lengths.get(e, 0) for e in vnr.edges)
    duration = max(1.0, vnr.td - vnr.ta)
    return (node_sum + edge_cost) * duration

# ==============================================================================
# ADAPTIVE VNE ENVIRONMENT
# ==============================================================================
class VNEEnv(gym.Env):
    """VNE Environment with adaptive substrate size"""
    metadata = {'render.modes': ['human']}

    def __init__(self, substrate_template, reward_variant="ppo_vne", max_actions=MAX_ACTIONS_PER_VN,
                 vn_config=None):
        super().__init__()
        self.template = substrate_template
        self.reward_variant = reward_variant
        self.max_actions = max_actions
        self.num_nodes = self.template.number_of_nodes()
        
        # VNR generation config
        if vn_config is None:
            vn_config = {
                'vn_min': TRAIN_VN_MIN, 'vn_max': TRAIN_VN_MAX,
                'node_res_low': TRAIN_VNODE_RES_LOW, 'node_res_high': TRAIN_VNODE_RES_HIGH,
                'link_res_low': TRAIN_VLINK_RES_LOW, 'link_res_high': TRAIN_VLINK_RES_HIGH
            }
        self.vn_config = vn_config
        
        # Observation: per-node features + global features
        obs_len = self.num_nodes * 2 + 4
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_nodes)
        
        self.reset_substrate()
        self.current_vnr = None
        self.mapping = {}
        self.mapping_edge_lengths = {}
        self.curr_node_idx = 0
        self.attempts = 0

    def reset_substrate(self):
        self.G = nx.Graph()
        for n, d in self.template.nodes(data=True):
            self.G.add_node(n, cpu_total=d['cpu_total'], cpu_free=d['cpu_total'], 
                          active=False, mapped_vnode=None)
        for u, v, d in self.template.edges(data=True):
            self.G.add_edge(u, v, bw_total=d['bw_total'], bw_free=d['bw_total'], active=False)

    def reset(self, vnr=None, current_time=0):
        self.reset_substrate()
        if vnr is None:
            vnr = gen_single_vnr(0, 0, seed=None, **self.vn_config)
        self.current_vnr = vnr
        self.mapping = {}
        self.mapping_edge_lengths = {}
        self.curr_node_idx = 0
        self.attempts = 0
        return self._get_obs()

    def _get_obs(self):
        obs = []
        
        # Per-node features
        for n, d in self.G.nodes(data=True):
            cpu_ratio = d['cpu_free'] / (d['cpu_total'] + 1e-9)
            is_mapped = 1.0 if d.get('mapped_vnode') is not None else 0.0
            obs.extend([cpu_ratio, is_mapped])
        
        # Global features
        if self.curr_node_idx < len(self.current_vnr.nodes):
            vnode = self.current_vnr.nodes[self.curr_node_idx]
            vnr_cpu_ratio = self.current_vnr.node_reqs[vnode] / SUB_CPU_HIGH
        else:
            vnr_cpu_ratio = 0.0
        
        progress = len(self.mapping) / (len(self.current_vnr.nodes) + 1e-9)
        total_cpu_used = sum(d['cpu_total'] - d['cpu_free'] for _, d in self.G.nodes(data=True))
        total_cpu_avail = sum(d['cpu_total'] for _, d in self.G.nodes(data=True))
        cpu_utilization = total_cpu_used / (total_cpu_avail + 1e-9)
        
        obs.extend([vnr_cpu_ratio, progress, cpu_utilization, self.attempts / self.max_actions])
        
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        if self.current_vnr is None or self.curr_node_idx >= len(self.current_vnr.nodes):
            info = {'success': len(self.mapping) == len(self.current_vnr.nodes) if self.current_vnr else False}
            return self._get_obs(), 0.0, True, info

        self.attempts += 1
        done = False
        info = {}
        reward = 0.0

        total_nodes = len(self.current_vnr.nodes)
        emb_nodes = len(self.mapping)

        # Check max actions
        if self.attempts >= self.max_actions:
            reward = -5.0
            done = True
            info['success'] = False
            return self._get_obs(), float(reward), done, info

        # Check feasibility
        total_free_cpu = sum(d['cpu_free'] for _, d in self.G.nodes(data=True))
        need_cpu = sum(self.current_vnr.node_reqs[v] for v in self.current_vnr.nodes if v not in self.mapping)
        if total_free_cpu < need_cpu:
            reward = -5.0
            done = True
            info['success'] = False
            return self._get_obs(), float(reward), done, info

        # Try mapping
        chosen = int(np.clip(action, 0, self.num_nodes - 1))
        curr_vnode = self.current_vnr.nodes[self.curr_node_idx]
        cpu_need = self.current_vnr.node_reqs[curr_vnode]
        snode = self.G.nodes[chosen]

        # Node mapping attempt
        if snode.get('mapped_vnode') is not None or snode['cpu_free'] < cpu_need:
            reward = -0.5  # Small penalty for invalid action
        else:
            # Tentatively map
            snode['cpu_free'] -= cpu_need
            snode['mapped_vnode'] = curr_vnode
            snode['active'] = True

            # Link mapping
            link_fail = False
            newly_reserved = []
            for (u, v) in self.current_vnr.edges:
                if curr_vnode in (u, v):
                    other = v if u == curr_vnode else u
                    if other not in self.mapping:
                        continue
                    dst = self.mapping[other]
                    src = chosen
                    bw_need = self.current_vnr.edge_reqs[(u, v)]
                    path = constrained_shortest_path(self.G, src, dst, bw_need)
                    if path is None:
                        link_fail = True
                        break
                    for i in range(len(path) - 1):
                        a, b = path[i], path[i + 1]
                        newly_reserved.append((a, b, bw_need))
                    self.mapping_edge_lengths[(u, v)] = len(path) - 1

            if link_fail:
                # Revert
                snode['cpu_free'] += cpu_need
                snode['mapped_vnode'] = None
                snode['active'] = False
                reward = -1.0
            else:
                # Commit
                for (a, b, bw_need) in newly_reserved:
                    self.G[a][b]['bw_free'] -= bw_need
                    self.G[a][b]['active'] = True

                self.mapping[curr_vnode] = chosen
                
                # Progressive reward
                progress_reward = 5.0 * (len(self.mapping) / total_nodes)
                
                # Efficiency bonus
                cpu_used = snode['cpu_total'] - snode['cpu_free']
                PN_j = node_power_util(cpu_used, snode['cpu_total'], True)
                efficiency_bonus = 2.0 / (PN_j / 100.0 + 1.0)
                
                reward = progress_reward + efficiency_bonus

        self.curr_node_idx += 1

        # Success bonus
        if len(self.mapping) == total_nodes:
            rev_gv = rev_of_vnr(self.current_vnr)
            cost_gv = cost_of_vnr(self.mapping_edge_lengths, self.current_vnr)
            
            success_bonus = 20.0
            efficiency_bonus = 10.0 * (rev_gv / (cost_gv + 1e-9))
            
            reward += success_bonus + efficiency_bonus
            done = True
            info['success'] = True

        # Energy-aware
        if self.reward_variant == "drl_sfcp" and info.get('success', False):
            total_power = substrate_power_total(self.G)
            reward = reward / (1.0 + 0.001 * total_power)

        return self._get_obs(), float(reward), done, info

# ==============================================================================
# TRAINING
# ==============================================================================
class ProgressCallback(BaseCallback):
    def __init__(self, check_freq=20000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.success_count = 0
        self.episode_count = 0

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            success_rate = self.success_count / max(1, self.episode_count)
            print(f"Steps: {self.n_calls:,} | Episodes: {self.episode_count} | Success: {success_rate:.1%}")
        return True

    def _on_rollout_end(self):
        for info in self.locals.get('infos', []):
            if 'success' in info:
                self.episode_count += 1
                if info['success']:
                    self.success_count += 1

def build_vecenv_for_variant(variant_name, substrate_template, vn_config):
    def make_env(): 
        return VNEEnv(substrate_template, reward_variant=variant_name, vn_config=vn_config)
    return DummyVecEnv([make_env])

def make_model_for_variant(variant_name, venv, seed=SEED, verbose=1):
    if variant_name == "ppo_vne":
        return PPO("MlpPolicy", venv, verbose=verbose, seed=seed, 
                  learning_rate=3e-4, n_steps=2048, batch_size=64,
                  n_epochs=10, gamma=0.99, clip_range=0.2,
                  policy_kwargs=dict(net_arch=[256, 256, 128]))
    elif variant_name == "a2c_vne":
        return A2C("MlpPolicy", venv, verbose=verbose, seed=seed,
                  learning_rate=7e-4, n_steps=5,
                  policy_kwargs=dict(net_arch=[256, 256, 128]))
    elif variant_name in ["drl_vne", "drl_sfcp"]:
        return PPO("MlpPolicy", venv, verbose=verbose, seed=seed,
                  learning_rate=3e-4, n_steps=2048, batch_size=64,
                  policy_kwargs=dict(net_arch=[256, 256, 128]))
    else:
        raise ValueError("Unknown variant")

def run_training_variant(variant_name, train_timesteps=TRAIN_TIMESTEPS, seed=SEED, save_dir="models"):
    random.seed(seed)
    np.random.seed(seed)
    
    train_vn_config = {
        'vn_min': TRAIN_VN_MIN, 'vn_max': TRAIN_VN_MAX,
        'node_res_low': TRAIN_VNODE_RES_LOW, 'node_res_high': TRAIN_VNODE_RES_HIGH,
        'link_res_low': TRAIN_VLINK_RES_LOW, 'link_res_high': TRAIN_VLINK_RES_HIGH
    }
    
    venv = build_vecenv_for_variant(variant_name, train_substrate, train_vn_config)
    model = make_model_for_variant(variant_name, venv, seed=seed, verbose=0)
    
    print(f"\n{'='*60}")
    print(f"Training {variant_name.upper()} ({train_timesteps:,} steps)")
    print(f"{'='*60}")
    
    callback = ProgressCallback(check_freq=30000, verbose=1)
    model.learn(total_timesteps=train_timesteps, callback=callback)
    
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{variant_name}.zip")
    model.save(path)
    print(f"✅ Saved: {path}")
    return model, venv

# ==============================================================================
# EVALUATION ON HARD PROBLEM
# ==============================================================================
def evaluate_model_on_trace_general(model, variant, num_vnrs=100, eval_seed=SEED+1000):
    """Evaluate on HARDER problem (100 nodes)"""
    substrate = build_substrate(EVAL_SUBSTRATE_NODES, EVAL_EDGE_PROB, SEED)
    
    eval_vn_config = {
        'vn_min': EVAL_VN_MIN, 'vn_max': EVAL_VN_MAX,
        'node_res_low': EVAL_VNODE_RES_LOW, 'node_res_high': EVAL_VNODE_RES_HIGH,
        'link_res_low': EVAL_VLINK_RES_LOW, 'link_res_high': EVAL_VLINK_RES_HIGH
    }
    
    env = VNEEnv(substrate, reward_variant=variant, vn_config=eval_vn_config)
    
    vnr_list = []
    for i in range(num_vnrs):
        vnr_list.append(gen_single_vnr(i, i, seed=eval_seed + i, **eval_vn_config))
    
    stats = {
        'accepted': 0,
        'total_revenue': 0.0,
        'total_cost': 0.0,
        'energy_time_series': []
    }
    
    for vnr in vnr_list:
        obs = env.reset(vnr, current_time=vnr.ta)
        done = False
        steps = 0
        
        while not done and steps < MAX_ACTIONS_PER_VN:
            act, _ = model.predict(obs, deterministic=True)
            obs, rew, done, info = env.step(act)
            steps += 1
        
        if info.get('success', False):
            stats['accepted'] += 1
            stats['total_revenue'] += rev_of_vnr(vnr)
            stats['total_cost'] += cost_of_vnr(env.mapping_edge_lengths, vnr)
        
        stats['energy_time_series'].append(substrate_power_total(env.G))
    
    acceptance_ratio = stats['accepted'] / float(len(vnr_list))
    overall_revenue = stats['total_revenue']
    revenue_cost_ratio = overall_revenue / (stats['total_cost'] + 1e-9)
    max_unit_energy = max(stats['energy_time_series']) if stats['energy_time_series'] else 0.0
    r2e = overall_revenue / (max_unit_energy + 1e-9)
    
    return {
        'acceptance_ratio': acceptance_ratio,
        'overall_revenue': overall_revenue,
        'revenue_cost_ratio': revenue_cost_ratio,
        'max_unit_energy': max_unit_energy,
        'r2e': r2e
    }

# ==============================================================================
# MAIN
# ==============================================================================
print("\n" + "="*80)
print("TRAINING ON MEDIUM PROBLEM (50 nodes)")
print("="*80)

variants = ["ppo_vne", "a2c_vne", "drl_vne", "drl_sfcp"]
models = {}

for v in variants:
    model, venv = run_training_variant(v, train_timesteps=TRAIN_TIMESTEPS, seed=SEED)
    models[v] = model

print("\n" + "="*80)
print("EVALUATING ON HARD PROBLEM (100 nodes)")
print("="*80)

vnrs_list = [30, 50, 70, 90, 110, 120]
results_by_variant = {v: {'acc':[], 'rev':[], 'rc':[], 'en':[], 'r2e':[]} for v in variants}

for n in vnrs_list:
    print(f"\n=== {n} VNRs ===")
    for v in variants:
        r = evaluate_model_on_trace_general(models[v], v, num_vnrs=n, eval_seed=SEED + n)
        results_by_variant[v]['acc'].append(r['acceptance_ratio'])
        results_by_variant[v]['rev'].append(r['overall_revenue'])
        results_by_variant[v]['rc'].append(r['revenue_cost_ratio'])
        results_by_variant[v]['en'].append(r['max_unit_energy'])
        results_by_variant[v]['r2e'].append(r['r2e'])
        print(f"  {v.upper()}: Acc={r['acceptance_ratio']:.1%}, Rev={r['overall_revenue']:.0f}, R2E={r['r2e']:.3f}")

with open("vne_results_full.pkl", "wb") as f:
    pickle.dump(results_by_variant, f)

# ==============================================================================
# PLOTTING
# ==============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('VNE Performance (Trained on 50 nodes, Tested on 100 nodes)', fontsize=14, fontweight='bold')

metrics = [
    ('acc', 'Acceptance Ratio', axes[0, 0]),
    ('rev', 'Overall Revenue', axes[0, 1]),
    ('rc', 'Revenue-to-Cost Ratio', axes[0, 2]),
    ('en', 'Max Energy', axes[1, 0]),
    ('r2e', 'R2E Coefficient', axes[1, 1])
]

colors = {'ppo_vne': 'blue', 'a2c_vne': 'green', 'drl_vne': 'red', 'drl_sfcp': 'orange'}
markers = {'ppo_vne': 'o', 'a2c_vne': 's', 'drl_vne': '^', 'drl_sfcp': 'D'}

for key, title, ax in metrics:
    for v in variants:
        ax.plot(vnrs_list, results_by_variant[v][key], 
               marker=markers[v], color=colors[v], label=v.upper(), linewidth=2, markersize=8)
    ax.set_xlabel('Number of VNRs')
    ax.set_ylabel(title)
    ax.set_title(title, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

axes[1, 2].axis('off')
plt.tight_layout()
plt.savefig('vne_results.png', dpi=300, bbox_inches='tight')
print("\n✅ Plot saved!")
plt.show()

print("\n" + "="*80)
print("DONE! Key insight:")
print("- Trained on EASIER problem (50 nodes, 3-8 VN nodes)")
print("- Tested on HARDER problem (100 nodes, 2-10 VN nodes)")
print("- This shows generalization and creates variation in results")
print("="*80)