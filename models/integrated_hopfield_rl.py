
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import argparse
import os
import imageio

# Define paths relative to user home to ensure they work in WSL/Linux correctly
DEFAULT_LOG_DIR = os.path.expanduser("~/Projects/PIML/logs")
DEFAULT_VIDEO_DIR = os.path.expanduser("~/Projects/PIML/videos")

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Hopfield Energy Net (Adapted) ---
class HopfieldEnergyNet(nn.Module):
    def __init__(self, num_neurons=1000):
        super().__init__()
        self.num_neurons = num_neurons
        self.norm = nn.LayerNorm(num_neurons)
        # Weights for internal energy interactions
        self.weights = nn.Parameter(torch.zeros(num_neurons, num_neurons))
        self.bias = nn.Parameter(torch.zeros(num_neurons))
        nn.init.xavier_uniform_(self.weights)
        
        # We'll use a linear projection for the 'readout' or 'value' prediction in MNIST
        # For RL, this might be used differently or ignored in favor of the extractor output.
        self.output_proj = nn.Linear(1, 1)

        # Register positions and distances as buffers (fixed topology)
        # We retain the grid structure from the original code (28x28 + extras)
        positions = torch.zeros(num_neurons, 2)
        # First 784 neurons correspond to 28x28 grid
        positions[:784] = torch.tensor([[i % 28, i // 28] for i in range(784)], dtype=torch.float32)
        # Remaining neurons placed arbitrarily (e.g., in a column)
        remaining = num_neurons - 784
        if remaining > 0:
             positions[784:] = torch.tensor([[28, i % 10] for i in range(remaining)], dtype=torch.float32)
        
        self.register_buffer("positions", positions)
        distances = torch.cdist(self.positions, self.positions)
        self.register_buffer("distances", distances)

    def energy(self, states):
        # E = -0.5 * s^T W s - b^T s
        interaction = torch.bmm(states.unsqueeze(1), torch.matmul(states, self.weights.transpose(-2, -1)).unsqueeze(2)).squeeze()
        bias_term = torch.matmul(states, self.bias)
        return -0.5 * interaction.mean() - bias_term.mean()

    def forward(self, input_indices, input_values, output_indices=None, steps=20, beta=1.0):
        """
        Args:
            input_indices: List or Tensor of indices of neurons to clamp/force with input.
            input_values: Tensor of shape (batch, len(input_indices)) containing values.
            output_indices: List of indices to read out from. If None, return all states.
            steps: Number of energy descent/update steps.
            beta: Inverse temperature for tanh.
        """
        batch_size = input_values.size(0)
        
        # Initialize internal states (can be random or zeros)
        states = torch.zeros(batch_size, self.num_neurons, device=self.weights.device)
        
        # We construct a mask or directly add inputs. 
        # For this architecture, let's assume inputs are added as an external field 
        # or clamped. The original code did: activation = ... + input_signal
        
        input_signal = torch.zeros(batch_size, self.num_neurons, device=self.weights.device)
        # input_indices might be a list
        if isinstance(input_indices, list):
             input_indices = torch.tensor(input_indices, device=self.weights.device)
        
        # Scatter inputs into the signal vector
        # input_values: (BxN_in) -> scatter to (BxTotal)
        input_signal.scatter_(1, input_indices.unsqueeze(0).expand(batch_size, -1), input_values)

        for _ in range(steps):
            activation = torch.matmul(states, self.weights) + self.bias + input_signal
            new_states = torch.tanh(beta * activation)
            states = self.norm(states + new_states)
            # Noise for exploration/stochasticity
            states = states + 0.01 * torch.randn_like(states)
            
        if output_indices is not None:
            # If we want specific outputs (like for the MNIST class prediction in the original code)
            # The original code projected *specific* neuron states to 1D via output_proj
            # Here we might just return the states if output_indices is special
            if isinstance(output_indices, list):
                 output_indices = torch.tensor(output_indices, device=self.weights.device)
            selected_states = states.index_select(1, output_indices)
            
            # Original MNIST logic: project selected neurons to 1 value? 
            # In original: self.output_proj(states[:, output_neurons].unsqueeze(-1)).squeeze(-1)
            # which implies output_neurons was a list, and it projected each independently?
            # actually it's likely (Batch, NumOutputNeurons, 1) -> (Batch, NumOutputNeurons)
            outputs = self.output_proj(selected_states.unsqueeze(-1)).squeeze(-1)
            return outputs, states
        
        return states

    def distance_penalty(self, lambda_dist=0.00005):
        return lambda_dist * (self.weights.abs() * self.distances).sum()

# --- Wrapper for Stable Baselines3 ---
class HopfieldFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, num_neurons=1000, steps=10):
        # The output features will be the state of all neurons (or a subset).
        # Let's return the full state for now to let the policy head decide.
        features_dim = num_neurons
        super(HopfieldFeatureExtractor, self).__init__(observation_space, features_dim)
        
        self.net = HopfieldEnergyNet(num_neurons=num_neurons)
        self.steps = steps
        
        # Define which neurons correspond to observations.
        # Ant-v4 observations are usually size 27.
        # We map them to the first 27 neurons (or scattered).
        n_input = observation_space.shape[0]
        self.input_indices = list(range(794, n_input + 794))
        
    def forward(self, observations):
        # observations shape: (Batch, ObsDim)
        # Call the Hopfield Net
        # We don't specify output_indices so we get the full state back
        states = self.net(self.input_indices, observations, output_indices=None, steps=self.steps)
        return states

# --- MNIST Training Function ---
def train_mnist():
    print("--- Starting MNIST Training ---")
    num_neurons = 1000
    # MNIST 28x28 = 784 pixels
    input_indices = list(range(784))
    # Digits 0-9 predicted by 10 output neurons? Or regression? 
    # Original code used output_neurons = list(range(784, 794)) for 10 classes
    output_indices = list(range(784, 794))
    
    model = HopfieldEnergyNet(num_neurons).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # Flatten image
        data = data.view(data.size(0), -1) 
        
        # Forward pass
        # outputs shape: (Batch, 10)
        outputs, _ = model(input_indices, data, output_indices=output_indices)
        
        # Cross Entropy Loss
        loss = F.cross_entropy(outputs, target)
        
        # Regularization
        sparsity_loss = 0.001 * model.weights.abs().sum()
        dist_penalty = model.distance_penalty(lambda_dist=0.0001)
        
        full_loss = loss + sparsity_loss + dist_penalty
        full_loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"MNIST Train Batch {batch_idx}: Loss {full_loss.item():.4f}")
        
        # if batch_idx > 300: # Short run for demo
        #     break

    print("MNIST Training Completed.\n")
    return model

# --- RL Training Function ---
def train_rl(pretrained_net=None):
    print("--- Starting RL (Ant) Training ---")
    
    # ensure log dir exists
    os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
    
    # Create Environment
    # We use DummyVecEnv for compatibility
    env = gym.make("Ant-v4", render_mode="rgb_array")
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Initialize PPO with Custom Policy
    # We pass the class definition to policy_kwargs
    policy_kwargs = dict(
        features_extractor_class=HopfieldFeatureExtractor,
        features_extractor_kwargs=dict(num_neurons=1000, steps=5), # Fewer steps for speed in RL
        net_arch=[dict(pi=[64, 64], vf=[64, 64])] # Small heads on top of the big feature extractor
    )
    
    log_path = os.path.join(DEFAULT_LOG_DIR, "ppo_ant_hopfield")
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, device=device, tensorboard_log=log_path)
    
    # Check if a pretrained network was provided and load its weights
    if pretrained_net is not None:
        print("Loading pretrained weights from provided HopfieldEnergyNet...")
        # Access the feature extractor within the policy
        # The structure is model.policy.features_extractor.net (as defined in HopfieldFeatureExtractor)
        model.policy.features_extractor.net.load_state_dict(pretrained_net.state_dict())

    
    # Train for a short duration to verify it runs and learns something
    model.learn(total_timesteps=5000)
    
    print("RL Training Finished.")
    
    # --- Visualization ---
    print("Generating Replay...")
    env_test = gym.make("Ant-v4", render_mode="rgb_array")
    obs, _ = env_test.reset()
    
    frames = []
    for _ in range(1000):
        # We need to normalize observation if we used VecNormalize, but for a quick check raw might be okay-ish 
        # or we should load stats. For simplicity in this demo, just run.
        # SB3 model.predict expects shape (d,) or (1, d)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env_test.step(action)
        frames.append(env_test.render())
        if terminated or truncated:
            obs, _ = env_test.reset()
            
    env_test.close()
    
    # Save video artifact
    if len(frames) > 0:
        os.makedirs(DEFAULT_VIDEO_DIR, exist_ok=True)
        video_path = os.path.join(DEFAULT_VIDEO_DIR, "ant_hopfield_replay.mp4")
        imageio.mimsave(video_path, frames, fps=30)
        print(f"Saved replay video to {video_path}")
        
    # Return the trained custom network (feature extractor)
    return model.policy.features_extractor.net

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="all", choices=["mnist", "rl", "all"])
    args = parser.parse_args()
    
    if args.mode in ["mnist", "all"]:
        train_mnist()
    
    if args.mode in ["rl", "all"]:
        train_rl()
