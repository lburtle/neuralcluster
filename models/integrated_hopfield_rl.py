
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

# --- Constants ---
NUM_NEURONS = 1500
# --- Constants ---
NUM_NEURONS = 1500
# V1-V5 Topology
# V1: 0-500 (Simple)
# V2: 500-900 (Complex/Associative)
# Ventral (MNIST): 900-1200
# Dorsal (RL): 1200-1500
V1_END = 500
V2_END = 900
VENTRAL_END = 1200
DORSAL_END = NUM_NEURONS

EWC_LAMBDA = 2000.0      # Importance of preserving old weights


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

        # Create Block-Diagonal Connectivity Mask
        self.create_connectivity_mask()

    def create_connectivity_mask(self):
        mask = torch.zeros(self.num_neurons, self.num_neurons)
        
        # Helper to set block values
        def set_block(start_r, end_r, start_c, end_c, value):
            mask[start_r:end_r, start_c:end_c] = value
            
        # 1. Intra-Layer Connectivity (Dense)
        set_block(0, V1_END, 0, V1_END, 1.0)              # V1 <-> V1
        set_block(V1_END, V2_END, V1_END, V2_END, 1.0)    # V2 <-> V2
        set_block(V2_END, VENTRAL_END, V2_END, VENTRAL_END, 1.0) # Ventral <-> Ventral
        set_block(VENTRAL_END, DORSAL_END, VENTRAL_END, DORSAL_END, 1.0) # Dorsal <-> Dorsal
        
        # 2. Feedforward / Feedback
        # V1 -> V2 (Strong)
        set_block(0, V1_END, V1_END, V2_END, 1.0)
        # V2 -> V1 (Weak Feedback)
        set_block(V1_END, V2_END, 0, V1_END, 0.1)
        
        # V2 -> Ventral (Strong)
        set_block(V1_END, V2_END, V2_END, VENTRAL_END, 1.0)
        # V2 -> Dorsal (Strong)
        set_block(V1_END, V2_END, VENTRAL_END, DORSAL_END, 1.0)
        
        # Ventral <-> Dorsal (Disconnected/Sparse)
        set_block(V2_END, VENTRAL_END, VENTRAL_END, DORSAL_END, 0.001)
        set_block(VENTRAL_END, DORSAL_END, V2_END, VENTRAL_END, 0.001)
        
        # 3. Global Workspace (Sparse Background)
        # Fill zeros with small value
        mask[mask == 0] = 0.005
        
        self.register_buffer("connectivity_mask", mask)

    def energy(self, states):
        # E = -0.5 * s^T (W * M) s - b^T s
        effective_weights = self.weights * self.connectivity_mask
        interaction = torch.bmm(states.unsqueeze(1), torch.matmul(states, effective_weights.transpose(-2, -1)).unsqueeze(2)).squeeze()
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
        # Scatter inputs into the signal vector
        # input_values: (BxN_in) -> scatter to (BxTotal)
        input_signal.scatter_(1, input_indices.unsqueeze(0).expand(batch_size, -1), input_values)

        for _ in range(steps):
            # Apply Connectivity Mask
            effective_weights = self.weights * self.connectivity_mask
            activation = torch.matmul(states, effective_weights) + self.bias + input_signal
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
    def __init__(self, observation_space: gym.spaces.Box, num_neurons=NUM_NEURONS, steps=10):
        n_input = observation_space.shape[0]
        # Map inputs to the Dorsal Stream input area? 
        # Or V1? RL agent usually sees "state".
        # Let's map inputs to V1 (0-n_input) for processing?
        # OR map to Dorsal Start?
        # User said: "Dorsal Stream... Specialized for RL".
        # But V1 processes "visual information". Ant state is "proprioception".
        # We will feed Ant state into V1 (0-...) to simulate "sensory input".
        self.input_indices = list(range(n_input)) 
        
        # OUTPUT: The policy should only see the DORSAL stream.
        # Dorsal range: VENTRAL_END to DORSAL_END (1200-1500)
        self.output_indices = list(range(VENTRAL_END, DORSAL_END))
        
        # Feature dim = size of Dorsal Stream
        features_dim = len(self.output_indices)
        super(HopfieldFeatureExtractor, self).__init__(observation_space, features_dim)
        
        self.net = HopfieldEnergyNet(num_neurons=num_neurons)
        self.steps = steps
        
    def forward(self, observations):
        # observations shape: (Batch, ObsDim)
        # Call the Hopfield Net
        # We pass output_indices to return only the hidden neuron states
        states, _ = self.net(self.input_indices, observations, output_indices=self.output_indices, steps=self.steps)
        return states

# --- MNIST Training Function ---
def train_mnist():
    print("--- Starting MNIST Training ---")
    # MNIST 28x28 = 784 pixels
    input_indices = list(range(784))
    # Digits 0-9 predicted by 10 output neurons? Or regression? 
    # Original code used output_neurons = list(range(784, 794)) for 10 classes
    output_indices = list(range(784, 794))
    
    # PARTITIONING: Only use the first MNIST_NEURONS_END neurons for this task.
    # The rest (1000-1500) are reserved for RL.
    # We enforce this by masking or just letting the optimizer handle it?
    # Better: We zero out gradients for the reserved neurons to ensure "Blank Slate"
    
    model = HopfieldEnergyNet(num_neurons=NUM_NEURONS).to(device)
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
        
        # Zero out gradients for reserved neurons (Partitioning)
        # Weights shape: (N, N), Bias shape: (N)
        # Zero out gradients for reserved neurons (Partitioning)
        # Weights shape: (N, N), Bias shape: (N)
        if model.weights.grad is not None:
            # Prevent updates to Dorsal Stream (RL) during MNIST training
            model.weights.grad[VENTRAL_END:, :] = 0
            model.weights.grad[:, VENTRAL_END:] = 0
        if model.bias.grad is not None:
            model.bias.grad[VENTRAL_END:] = 0
            
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"MNIST Train Batch {batch_idx}: Loss {full_loss.item():.4f}")
        
        # if batch_idx > 300: # Short run for demo
        #     break

    print("MNIST Training Completed.\n")
    return model

def compute_fisher(model, dataloader):
    print("Computing Fisher Information Matrix for EWC...")
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    model.eval()
    
    # We only care about the weights potentially used in MNIST
    input_indices = list(range(784))
    output_indices = list(range(784, 794))
    
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)
        model.zero_grad()
        outputs, _ = model(input_indices, data, output_indices=output_indices)
        loss = F.cross_entropy(outputs, target)
        loss.backward()
        
        for n, p in model.named_parameters():
             if p.requires_grad and p.grad is not None:
                 fisher[n] += p.grad.data ** 2
                 
    # Normalize
    for n in fisher:
        fisher[n] /= len(dataloader)
        
    return fisher

# --- RL Training Function ---
def train_rl(pretrained_net=None):
    # REMOVED freeze_features argument per user request
    # Instead we use EWC + Partitioning
    
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
        features_extractor_kwargs=dict(num_neurons=NUM_NEURONS, steps=5), # Fewer steps for speed in RL
        net_arch=[dict(pi=[64, 64], vf=[64, 64])] # Small heads on top of the big feature extractor
    )
    
    log_path = os.path.join(DEFAULT_LOG_DIR, "ppo_ant_hopfield")
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, device=device, tensorboard_log=log_path)
    
    # Check if a pretrained network was provided and load its weights
    if pretrained_net is not None:
        print("Loading pretrained weights from provided HopfieldEnergyNet...")
        model.policy.features_extractor.net.load_state_dict(pretrained_net.state_dict())
        
        # EWC Setup
        print("Calculating EWC importance weights...")
        # Re-create dataloader for Fisher calc (quick hack, ideally passed in)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        # small subset for speed
        subset = torch.utils.data.Subset(dset, indices=range(1000))
        loader = torch.utils.data.DataLoader(subset, batch_size=64)
        
        fisher_matrix = compute_fisher(model.policy.features_extractor.net, loader)
        # Store old parameters
        old_params = {n: p.clone().detach() for n, p in model.policy.features_extractor.net.named_parameters() if p.requires_grad}
        
    # --- Custom Optimizer Wrapper for EWC ---
    # We hook into the optimizer to add the EWC penalty step
    # This is a bit tricky with SB3. 
    # Instead, we will perform a 'post-step' hook using a callback or just ensure the optimizer minimizes the combined loss?
    # SB3 doesn't easily allow checking custom loss components.
    # Workaround: A custom torch optimizer that adds the gradient of the EWC penalty to the existing gradients.
    
    class EWCOptimizer(torch.optim.Adam):
        def step(self, closure=None):
            # Calculate EWC gradients manually and add to p.grad
            if pretrained_net is not None:
                # EWC Loss = lambda * sum(fisher * (p - old)^2)
                # Gradient of EWC Loss = 2 * lambda * fisher * (p - old)
                net = model.policy.features_extractor.net
                for n, p in net.named_parameters():
                    if n in fisher_matrix and p.grad is not None:
                        ewc_grad = 2 * EWC_LAMBDA * fisher_matrix[n] * (p.data - old_params[n])
                        p.grad.data += ewc_grad
                        
            super().step(closure)

    # Hack: Inject our custom optimizer class into PPO
    # PPO uses policy_kwargs['optimizer_class'] if provided, or defaults to Adam.
    # But we need to initialize it with our captured variables (fisher, old_params).
    # Since we can't pickle local classes easily for multiprocessing (if used), we rely on DummyVecEnv (single process).
    
    # Actually, SB3 allows passing 'policy_kwargs' dict. 
    # We can pass the *class*, but not the instance.
    # So we define the class *inside* train_rl so it captures the closure, 
    # AND we must monkey-patch it or pass it as 'optimizer_class'.
    # Note: SB3 creates the optimizer internally using `self.policy_kwargs.get("optimizer_class", th.optim.Adam)`
    
    # Let's overwrite the policy's optimizer AFTER initialization? 
    # PPO initializes optimizer in `_setup_model`.
    # We can replace `model.policy.optimizer` with our own wrapper?
    # Yes, `model.policy.optimizer` is a standard torch optimizer.
    
    # 1. Initialize PPO normally (with standard Adam)
    # model = PPO(...) # Already done above
    
    # 2. Re-create the optimizer using our custom EWC logic
    if pretrained_net is not None:
        params = list(model.policy.parameters())
        # We need to use the exact same params (learning rate etc)
        # PPO default lr is 3e-4
        model.policy.optimizer = EWCOptimizer(params, lr=3e-4) # Replaces the standard Adam
        print("Replaced PPO optimizer with custom EWC+Adam optimizer.")

    
    # Train for a short duration to verify it runs and learns something

    
    # Train for a short duration to verify it runs and learns something
    model.learn(total_timesteps=12000)
    
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
    # Save the model
    model_save_path = os.path.join(DEFAULT_LOG_DIR, "hopfield_rl_model.pt")
    torch.save(model.policy.features_extractor.net.state_dict(), model_save_path)
    print(f"Saved trained HopfieldEnergyNet to {model_save_path}")

    return model.policy.features_extractor.net

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="all", choices=["mnist", "rl", "all"])
    args = parser.parse_args()
    
    mnist_model = None
    if args.mode in ["mnist", "all"]:
        mnist_model = train_mnist()
    
    if args.mode in ["rl", "all"]:
        train_rl(pretrained_net=mnist_model)
