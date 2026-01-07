import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
import argparse
import os
import imageio
from tqdm.auto import tqdm

# Define paths relative to user home to ensure they work in WSL/Linux correctly
DEFAULT_LOG_DIR = os.path.expanduser("~/Projects/PIML/logs")
DEFAULT_VIDEO_DIR = os.path.expanduser("~/Projects/PIML/videos")

# --- Constants ---
NUM_NEURONS = 3500
# V1-V5 Topology
# V1: 0-1000 (Simple)
# V2: 1000-1800 (Complex/Associative)
# Ventral (MNIST): 1800-2300
# Dorsal (RL): 2300-3500
V1_END = 1000
V2_END = 1800
VENTRAL_END = 2300
DORSAL_END = NUM_NEURONS

# --- SURVIVAL SETTING: High Protection for Vision ---
EWC_LAMBDA = 5000.0     # Increased from 2000 to 5000 to prevent "Blindness"

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Hopfield Energy Net (Adapted) ---
class HopfieldEnergyNet(nn.Module):
    def __init__(self, num_neurons=NUM_NEURONS):
        super().__init__()
        self.num_neurons = num_neurons
        self.norm = nn.LayerNorm(num_neurons)
        
        # Weights for internal energy interactions
        self.weights = nn.Parameter(torch.empty(num_neurons, num_neurons))
        self.bias = nn.Parameter(torch.zeros(num_neurons))
        
        # Centered Normal Distribution for Excitatory/Inhibitory Balance
        nn.init.normal_(self.weights, mean=0.0, std=0.02)
        
        self.output_proj = nn.Linear(1, 1)

        # Register positions and distances as buffers (fixed topology)
        positions = torch.zeros(num_neurons, 2)
        positions[:784] = torch.tensor([[i % 28, i // 28] for i in range(784)], dtype=torch.float32)
        remaining = num_neurons - 784
        if remaining > 0:
             positions[784:] = torch.tensor([[28, i % 10] for i in range(remaining)], dtype=torch.float32)
        
        self.register_buffer("positions", positions)
        distances = torch.cdist(self.positions, self.positions)
        self.register_buffer("distances", distances)

        # Create Block-Diagonal Connectivity Mask
        self.create_connectivity_mask()

    def create_connectivity_mask(self):
        mask = torch.zeros(self.num_neurons, self.num_neurons)
        
        def set_block(start_r, end_r, start_c, end_c, value):
            mask[start_r:end_r, start_c:end_c] = value

        # --- 1. INTRA-LAYER (Internal Stability) ---
        set_block(0, V1_END, 0, V1_END, 1.0)                      # V1: Raw Storage
        set_block(V1_END, V2_END, V1_END, V2_END, 0.8)            # V2: Feature Association
        set_block(V2_END, VENTRAL_END, V2_END, VENTRAL_END, 1.0)  # Ventral: Identity (MNIST)
        set_block(VENTRAL_END, DORSAL_END, VENTRAL_END, DORSAL_END, 0.3) # Dorsal: Motor (RL)

        # --- 2. THE VISION STREAM (What) ---
        set_block(0, V1_END, V1_END, V2_END, 1.0)                 # V1 -> V2 (Feedforward)
        set_block(V1_END, V2_END, V2_END, VENTRAL_END, 1.0)       # V2 -> Ventral (Recognition)
        set_block(VENTRAL_END, V2_END, V1_END, VENTRAL_END, 0.2)  # Ventral -> V2 (Feedback/Attention)

        # --- 3. THE MOTOR STREAM (How) ---
        # SURVIVAL SETTING: The Muzzle (0.07)
        # Prevents Proprioceptive Noise from drowning out the Vision Signal
        set_block(V1_END, V2_END, VENTRAL_END, DORSAL_END, 0.07)   

        # --- 4. THE INTEGRATION BRIDGE (The "Even/Odd" Rule) ---
        # SURVIVAL SETTING: The Amplifier (0.7)
        # Makes the Vision Signal 10x louder than the Muzzled Joints
        set_block(V2_END, VENTRAL_END, VENTRAL_END, DORSAL_END, 0.7) 
        set_block(VENTRAL_END, DORSAL_END, V2_END, VENTRAL_END, 0.01) 

        # --- 5. GLOBAL WORKSPACE ---
        mask[mask == 0] = 0.0005 # Sparse background connectivity
    
        self.register_buffer("connectivity_mask", mask)

    def energy(self, states):
        effective_weights = self.weights * self.connectivity_mask
        interaction = torch.bmm(states.unsqueeze(1), torch.matmul(states, effective_weights.transpose(-2, -1)).unsqueeze(2)).squeeze()
        bias_term = torch.matmul(states, self.bias)
        return -0.5 * interaction.mean() - bias_term.mean()

    def forward(self, input_indices, input_values, output_indices=None, steps=20, beta_start=0.5, beta_end=2.0):
        batch_size = input_values.size(0)
        
        states = torch.zeros(batch_size, self.num_neurons, device=self.weights.device)
        input_signal = torch.zeros(batch_size, self.num_neurons, device=self.weights.device)
        
        if isinstance(input_indices, list):
             input_indices = torch.tensor(input_indices, device=self.weights.device)
        
        # Scatter inputs into the signal vector
        input_signal.scatter_(1, input_indices.unsqueeze(0).expand(batch_size, -1), input_values)

        for i in range(steps):
            # ANNEALING: Linearly increase beta from beta_start to beta_end
            # This helps the network explore (low beta) then commit (high beta)
            current_beta = beta_start + (i / steps) * (beta_end - beta_start)
            
            effective_weights = self.weights * self.connectivity_mask
            activation = torch.matmul(states, effective_weights) + self.bias + input_signal
            
            new_states = torch.tanh(current_beta * activation)
            states = self.norm(states + new_states)
            states = states + 0.01 * torch.randn_like(states)
            
        if output_indices is not None:
            if isinstance(output_indices, list):
                 output_indices = torch.tensor(output_indices, device=self.weights.device)
            selected_states = states.index_select(1, output_indices)
            outputs = self.output_proj(selected_states.unsqueeze(-1)).squeeze(-1)
            return outputs, states
        
        return states

    def distance_penalty(self, lambda_dist=0.00005):
        return lambda_dist * (self.weights.abs() * self.distances).sum()

# --- Wrapper for Stable Baselines3 ---
class HopfieldFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, num_neurons=NUM_NEURONS, steps=10):
        n_input = observation_space.shape[0] 
        
        self.input_indices = list(range(n_input)) 
        self.output_indices = list(range(VENTRAL_END, DORSAL_END))
        
        features_dim = len(self.output_indices)
        super(HopfieldFeatureExtractor, self).__init__(observation_space, features_dim)
        
        self.net = HopfieldEnergyNet(num_neurons=num_neurons)
        self.steps = steps
        
    def forward(self, observations):
        # We pass output_indices to return only the hidden neuron states
        states, _ = self.net(self.input_indices, observations, output_indices=self.output_indices, steps=self.steps)
        return states

## Injects MNIST images into environment for visual and locomotive association
class VisualAntWrapper(gym.Wrapper):
    def __init__(self, env, mnist_loader):
        super().__init__(env)
        self.mnist_loader = mnist_loader
        self.iterator = iter(mnist_loader)
        self.current_pixels = None
        self.current_label = None
        
        ant_obs_dim = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(ant_obs_dim + 784,)
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        try:
            self.current_pixels, self.current_label = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.mnist_loader)
            self.current_pixels, self.current_label = next(self.iterator)
        
        self.current_pixels = self.current_pixels.view(-1).numpy()
        return np.concatenate([obs, self.current_pixels]), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        forward_vel = obs[0]
        is_even = (self.current_label % 2 == 0)
        
        if is_even:
            # High reward for movement
            reward += forward_vel * 15.0 # Increased incentive to GO
        else:
            # SURVIVAL SETTING: The Slope (Continuous Gradient)
            # We provide a linear gradient so the agent feels "less pain" as it slows down.
            if forward_vel > 0:
                reward -= forward_vel * 100.0 # Linear penalty
            
            # Note: No termination needed. The gradient is the guide.

        full_obs = np.concatenate([obs, self.current_pixels])
        return full_obs, reward, terminated, truncated, info


def verify_architecture_integrity():
    print("\n" + "="*50)
    print("--- ARCHITECTURE INTEGRITY SANITY CHECK ---")
    print("="*50)
    
    blocks = [
        ("V1 (Sensory)", 0, V1_END),
        ("V2 (Associative)", V1_END, V2_END),
        ("Ventral (Vision)", V2_END, VENTRAL_END),
        ("Dorsal (Motor)", VENTRAL_END, DORSAL_END)
    ]
    
    for name, start, end in blocks:
        size = end - start
        print(f"-> {name:18} | Range: {start:4}-{end:4} | Size: {size:4}")

    ant_dim = 27
    mnist_dim = 784
    total_input = ant_dim + mnist_dim 
    
    print(f"\n-> Input Mapping:")
    print(f"   Ant Proprioception Port: Neurons 0 - {ant_dim-1}")
    print(f"   MNIST Vision Port:        Neurons {ant_dim} - {total_input-1}")
    
    if total_input > V1_END:
        print(f"   [!] WARNING: Inputs ({total_input}) exceed V1 capacity ({V1_END})!")
    else:
        print(f"   [v] Success: Inputs fit 1-to-1 inside V1")
    print("="*50 + "\n")

# --- MNIST Training Function ---
def train_mnist():
    print("--- Starting MNIST Training ---")
    input_indices = list(range(784))
    output_indices = list(range(VENTRAL_END - 10, VENTRAL_END))
    
    model = HopfieldEnergyNet(num_neurons=NUM_NEURONS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        data = data.view(data.size(0), -1) 
        
        outputs, _ = model(input_indices, data, output_indices=output_indices)
        loss = F.cross_entropy(outputs, target)
        
        # Regularization on Ventral block
        active_weights = model.weights[:VENTRAL_END, :VENTRAL_END]
        sparsity_loss = 0.00005 * active_weights.abs().sum()
        dist_penalty = model.distance_penalty(lambda_dist=0.00001) 

        full_loss = loss + sparsity_loss + dist_penalty
        full_loss.backward()
        
        # Zero out gradients for reserved neurons (Partitioning)
        if model.weights.grad is not None:
            model.weights.grad[VENTRAL_END:, :] = 0
            model.weights.grad[:, VENTRAL_END:] = 0
        if model.bias.grad is not None:
            model.bias.grad[VENTRAL_END:] = 0
            
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"MNIST Train Batch {batch_idx}: Loss {full_loss.item():.4f}")

    print("MNIST Training Completed.\n")
    return model

def compute_fisher(model, dataloader):
    print("Computing Fisher Information Matrix for EWC...")
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    model.eval()
    
    input_indices = list(range(784))
    output_indices = list(range(VENTRAL_END - 10, VENTRAL_END))
    
    # Calculate over entire dataset (or subset)
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)
        model.zero_grad()
        
        outputs, _ = model(input_indices, data, output_indices=output_indices, steps=20)
        loss = F.cross_entropy(outputs, target)
        loss.backward()
        
        for n, p in model.named_parameters():
             if p.requires_grad and p.grad is not None:
                 fisher[n] += p.grad.data ** 2
                 
    for n in fisher:
        fisher[n] /= len(dataloader)
        
    return fisher

class BridgeMonitor:
    def __init__(self, model, device, v_range=(1800, 2300), d_range=(2300, 3500)):
        self.model = model
        self.device = device
        self.v_range = v_range
        self.d_range = d_range
        self.pbar = None
        
        # Store Initial Weights for Plasticity Verification
        # (We detach and clone to keep a static reference)
        with torch.no_grad():
            self.initial_bridge_weights = self.model.weights[v_range[0]:v_range[1], 
                                                             d_range[0]:d_range[1]].detach().clone()
        
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset = datasets.MNIST('./data', train=False, transform=transform)
        self.even_img, self.odd_img = None, None
        for img, label in dataset:
            if label % 2 == 0 and self.even_img is None:
                self.even_img = img.view(1, -1).to(device)
            elif label % 2 != 0 and self.odd_img is None:
                self.odd_img = img.view(1, -1).to(device)
            if self.even_img is not None and self.odd_img is not None: break

    def check_health(self, step_count):
        self.model.eval()
        with torch.no_grad():
            vis_indices = list(range(27, 27 + 784))
            even_state = self.model(vis_indices, self.even_img, steps=30)
            odd_state = self.model(vis_indices, self.odd_img, steps=30)
            
            d_even = even_state[0, self.d_range[0]:self.d_range[1]]
            d_odd = odd_state[0, self.d_range[0]:self.d_range[1]]
            
            dist = torch.norm(d_even - d_odd, p=2).item()
            cos_sim = F.cosine_similarity(d_even.unsqueeze(0), d_odd.unsqueeze(0)).item()
            
            # --- DIAGNOSTIC: Check Bridge Magnitude ---
            bridge_w = self.model.weights[self.v_range[0]:self.v_range[1], 
                                          self.d_range[0]:self.d_range[1]]
            w_norm = torch.norm(bridge_w, p='fro').item()
            
            if self.pbar is not None:
                self.pbar.set_postfix({
                    "Dist": f"{dist:.2f}",
                    "Sim": f"{cos_sim:.2f}",
                    "W_Norm": f"{w_norm:.2f}"
                })
        
        print(f"\n[Bridge Check] Step {step_count}: Dist={dist:.4f}, Sim={cos_sim:.4f}, W_Norm={w_norm:.4f}")
        
        # --- Visualization: Bridge Heatmap ---
        self.visualize_bridge_heatmaps(step_count)

    def visualize_bridge_heatmaps(self, step_count):
        print("\n--- Generating Bridge Connectivity Heatmaps ---")
        self.model.eval()
        
        # Define block indices
        # Ventral (What): 1800-2300
        # Dorsal (Action): 2300-3500
        v_start, v_end = 1800, 2300
        d_start, d_end = 2300, 3500
        
        # Extract the bridge weights from the full weight matrix
        # We want weights WHERE Ventral is the input and Dorsal is the output
        with torch.no_grad():
            full_weights = self.model.weights * self.model.connectivity_mask
            bridge_weights = full_weights[v_start:v_end, d_start:d_end].cpu()
            
            # Use stored initial weights (on same device as when created, likely GPU)
            # Make sure to move to CPU for plotting
            current_w = bridge_weights.cpu()
            initial_w = self.initial_bridge_weights.cpu()
            
            diff_w = current_w - initial_w
            
            # --- DEBUG LOGGING: Drift ---
            drift_norm = torch.norm(diff_w, p='fro').item()
            print(f"[Heatmap Debug] Weight Drift (L2 Distance from Start): {drift_norm:.5f}")

        # 1. Plot Current State
        plt.figure(figsize=(10, 8))
        sns.heatmap(current_w.numpy(), cmap="RdBu_r", center=0)
        plt.title(f"Functional Bridge step {step_count}: Ventral (Y) to Dorsal (X)\nL2 Drift: {drift_norm:.4f}")
        plt.xlabel("Dorsal Neurons (Motor indices 2300-3500)")
        plt.ylabel("Ventral Neurons (Vision indices 1800-2300)")
        
        save_path = f"images/bridge_heatmap.png"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved bridge connectivity heatmap to {save_path}")
        
        # 2. Plot Difference (Plasticity)
        plt.figure(figsize=(10, 8))
        # Use a diverging map centered at 0 to show increase (Red) vs Decrease (Blue)
        # Robust scale: +/- 3 sigma of the diff
        limit = 3 * diff_w.std().item()
        sns.heatmap(diff_w.numpy(), cmap="RdBu_r", center=0, vmin=-limit, vmax=limit)
        plt.title(f"Synaptic Plasticity (Change from Start) - Step {step_count}")
        plt.xlabel("Dorsal Neurons (Motor)")
        plt.ylabel("Ventral Neurons (Vision)")
        
        diff_path = f"images/bridge_diff.png"
        plt.tight_layout()
        plt.savefig(diff_path)
        plt.close()
        print(f"Saved bridge DIFFERENCE heatmap to {diff_path}")

class HopfieldDiagnosticCallback(BaseCallback):
    def __init__(self, monitor, check_freq=5000): 
        super().__init__()
        self.monitor = monitor
        self.check_freq = check_freq

    def _on_training_start(self):
        if hasattr(self.training_env, "pbar"):
            self.monitor.pbar = self.training_env.pbar

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.monitor.check_health(self.num_timesteps)
        return True

class SchedulerCallback(BaseCallback):
    def __init__(self, optimizer, gamma=0.98, step_size=10000, verbose=1):
        super().__init__(verbose)
        self.optimizer = optimizer
        self.scheduler = ExponentialLR(self.optimizer, gamma=gamma)
        self.step_size = step_size

    def _on_step(self) -> bool:
        if self.n_calls % self.step_size == 0:
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.verbose > 0:
                print(f"\n[Scheduler] Step {self.num_timesteps}: Learning Rate reduced to {current_lr:.6f}")
        return True

# --- RL Training Function ---
def train_rl(pretrained_net=None):
    # Ensure log dir exists
    os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
    
    print("--- Starting RL (Integrated Visual Ant) Training ---")
    
    # 1. Load MNIST for the Wrapper
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_loader = torch.utils.data.DataLoader(mnist_train, batch_size=1, shuffle=True)

    # 2. Create and Wrap Environment
    raw_env = gym.make("Ant-v4", render_mode="rgb_array", terminate_when_unhealthy=False)
    # Apply our new Integration Wrapper with the "Death Penalty"
    visual_env = VisualAntWrapper(raw_env, mnist_loader)
    
    # Standard SB3 vectorization and normalization
    env = DummyVecEnv([lambda: visual_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Initialize PPO with Custom Policy
    policy_kwargs = dict(
        features_extractor_class=HopfieldFeatureExtractor,
        features_extractor_kwargs=dict(num_neurons=NUM_NEURONS, steps=5), # Fewer steps for speed in RL
        net_arch=[dict(pi=[64, 64], vf=[64, 64])] 
    )
    
    log_path = os.path.join(DEFAULT_LOG_DIR, "ppo_ant_hopfield")
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, device=device, tensorboard_log=log_path)
    
    # 3. Load Weights and Apply BIOLOGICAL GATING
    if pretrained_net is not None:
        print("Loading pretrained weights from provided HopfieldEnergyNet...")
        model.policy.features_extractor.net.load_state_dict(pretrained_net.state_dict())
        
        print("Engaging Dopamine Gating (Biological Compartmentalization)...")
        
        # --- THE DOPAMINE GATING HOOK ---
        # This hook intercepts the gradient signal from the RL algorithm.
        # It ensures that "Blame" for falling over stops at the bridge 
        # and does not scramble the Visual Cortex.
        
        def dopamine_gating_hook(grad):
            # Create a clone to modify
            grad_clone = grad.clone()
            
            # 1. FREEZE INTERNAL VISION (V1, V2, Ventral)
            # We zero out gradients for the square block of Vision-to-Vision connections.
            # Indices: 0 to VENTRAL_END (0-2300)
            grad_clone[:VENTRAL_END, :VENTRAL_END] = 0
            
            # 2. FREEZE FEEDBACK (Dorsal -> Vision)
            # We prevent the Motor block from "shouting back" and rewriting the Vision.
            # Rows: 0 to VENTRAL_END (Vision Neurons)
            # Cols: VENTRAL_END to DORSAL_END (Motor Neurons)
            grad_clone[:VENTRAL_END, VENTRAL_END:] = 0
            
            # 3. ALLOW BRIDGE UPDATES (Vision -> Dorsal)
            # We explicitly ALLOW gradients in the block where the Motor cortex 
            # "listens" to the Vision cortex. 
            # This is biologically accurate: The synapse exists on the Motor neuron,
            # so the Motor neuron is allowed to change the weight of that synapse.
            # Rows: VENTRAL_END to DORSAL_END (Motor Neurons receiving input)
            # Cols: 0 to VENTRAL_END (Vision Neurons sending output)
            # (No action needed, as we simply didn't zero this block)
            
            return grad_clone

        # Register the hook on the main weight matrix
        model.policy.features_extractor.net.weights.register_hook(dopamine_gating_hook)
        
        # Also freeze the Biases for the Vision Block
        def bias_gating_hook(grad):
            grad_clone = grad.clone()
            grad_clone[:VENTRAL_END] = 0 # Protect Vision Biases
            return grad_clone
            
        model.policy.features_extractor.net.bias.register_hook(bias_gating_hook)
        
        print("   -> Vision Cortex (0-2300) Protected.")
        print("   -> Synaptic Bridge (Vision->Motor) is PLASTIC (Learnable).")

    # 4. Create the Monitors and Callbacks
    monitor = BridgeMonitor(model.policy.features_extractor.net, device, 
                            v_range=(V2_END, VENTRAL_END), 
                            d_range=(VENTRAL_END, DORSAL_END))
    
    diag_callback = HopfieldDiagnosticCallback(monitor, check_freq=10000)
    
    # Scheduler: Cools down the Motor learning over time
    sched_callback = SchedulerCallback(model.policy.optimizer, gamma=0.97, step_size=10000)

    from stable_baselines3.common.callbacks import CallbackList
    callback = CallbackList([diag_callback, sched_callback])

    # 5. Train for Survival
    # We use 500k steps to ensure the 'Aha!' moment happens
    print("Training for 200,000 steps with Gating Active...")
    model.learn(total_timesteps=200000, callback=callback)
    
    print("RL Training Finished.")
    
    # --- Visualization ---
    import cv2 

    print("Generating Visualized Replay...")
    # Create a fresh test environment
    env_test = gym.make("Ant-v4", render_mode="rgb_array")
    env_test = VisualAntWrapper(env_test, mnist_loader)
    env_test = DummyVecEnv([lambda: env_test])
    
    # Sync Normalization stats
    env_test = VecNormalize(env_test, norm_obs=True, norm_reward=False, training=False)
    env_test.obs_rms = env.obs_rms 

    obs = env_test.reset()
    final_frames = [] 

    for i in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env_test.step(action)
        
        # 1. Grab the frame
        raw_frame = env_test.render()
        
        # 2. Check if the frame is valid/black
        if raw_frame is None or np.max(raw_frame) == 0:
            if i == 0: print("Warning: First frame is black. This is common in MuJoCo initialization.")
            continue

        # 3. Create a writeable, contiguous copy for OpenCV
        frame = np.ascontiguousarray(raw_frame.copy())
        
        # 4. DRAW THE DIGIT OVERLAY
        current_label = env_test.envs[0].current_label
        label_text = f"Digit: {current_label} ({'GO' if current_label % 2 == 0 else 'STOP'})"
        color = (0, 255, 0) if current_label % 2 == 0 else (0, 0, 255) 
        
        cv2.putText(frame, label_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # 5. Picture-in-Picture (PiP)
        mnist_vis = env_test.envs[0].current_pixels.reshape(28, 28)
        mnist_vis = (mnist_vis - mnist_vis.min()) / (mnist_vis.max() - mnist_vis.min()) * 255
        mnist_vis = cv2.resize(mnist_vis.astype(np.uint8), (150, 150))
        mnist_vis = cv2.cvtColor(mnist_vis, cv2.COLOR_GRAY2RGB)
        
        # Overlay on frame
        frame[10:160, -160:-10] = mnist_vis
        cv2.rectangle(frame, (-160, 10), (-10, 160), (255, 255, 255), 2)

        final_frames.append(frame)
            
    env_test.close()

    # Save video artifact
    if len(final_frames) > 0:
        os.makedirs(DEFAULT_VIDEO_DIR, exist_ok=True)
        video_path = os.path.join(DEFAULT_VIDEO_DIR, "ant_hopfield_replay.mp4")
        imageio.mimsave(video_path, final_frames, fps=30, macro_block_size=1)
        print(f"Successfully saved {len(final_frames)} frames to {video_path}")
    else:
        print("ERROR: No frames were captured. Replay video not saved.")
        
    # Return the trained custom network (feature extractor)
    model_save_path = os.path.join(DEFAULT_LOG_DIR, "hopfield_rl_model.pt")
    torch.save(model.policy.features_extractor.net.state_dict(), model_save_path)
    print(f"Saved trained HopfieldEnergyNet to {model_save_path}")

    return model.policy.features_extractor.net

if __name__ == "__main__":
    verify_architecture_integrity()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="all", choices=["mnist", "rl", "all"])
    args = parser.parse_args()
    
    mnist_model = None
    if args.mode in ["mnist", "all"]:
        mnist_model = train_mnist()
    
    if args.mode in ["rl", "all"]:
        train_rl(pretrained_net=mnist_model)