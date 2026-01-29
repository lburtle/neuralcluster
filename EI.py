import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os

## This allows us to see the E/I balance of the network
## and to see if the network is stable or not

# --- 1. SETUP (Use your existing class definition) ---
NUM_NEURONS = 3500
# Re-define the class structure briefly to load weights (or import from your file)
class HopfieldEnergyNet(nn.Module):
    def __init__(self, num_neurons=NUM_NEURONS):
        super().__init__()
        self.num_neurons = num_neurons
        self.weights = nn.Parameter(torch.empty(num_neurons, num_neurons))
        self.bias = nn.Parameter(torch.zeros(num_neurons))
        self.register_buffer("connectivity_mask", torch.ones(num_neurons, num_neurons)) # Placeholder
        # (We assume create_connectivity_mask is called if initializing fresh)

def analyze_brain_physics(model_path=None):
    print(f"--- ANALYZING NETWORK PHYSICS ({NUM_NEURONS} Neurons) ---")
    
    model = HopfieldEnergyNet()
    
    # Load weights if path provided, else use random initialization
    if model_path and os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        try:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        except:
            print("Could not load full state dict (keys might differ). Analyzing random init instead.")
            nn.init.normal_(model.weights, mean=0.0, std=0.02)
    else:
        print("No model found. Analyzing FRESH initialization (Normal Distribution).")
        nn.init.normal_(model.weights, mean=0.0, std=0.02)

    # Get weights as numpy array
    W = model.weights.detach().cpu().numpy()
    
    # --- 2. CHECK EXCITATORY / INHIBITORY BALANCE ---
    positive_neurons = np.sum(W > 0)
    negative_neurons = np.sum(W < 0)
    total_conns = W.size
    
    print(f"\n[Balance Check]")
    print(f"Excitatory Connections (+): {positive_neurons} ({positive_neurons/total_conns:.1%})")
    print(f"Inhibitory Connections (-): {negative_neurons} ({negative_neurons/total_conns:.1%})")
    
    if 0.49 < positive_neurons/total_conns < 0.51:
        print("-> STATUS: PERFECTLY BALANCED.")
    else:
        print("-> STATUS: IMBALANCED (Risk of Seizure or Silence)")

    # --- 3. CALCULATE SPECTRAL RADIUS ---
    print("\n[Spectral Radius Check]")
    print("Calculating Eigenvalues (this may take 10-20 seconds)...")
    
    # We calculate eigenvalues of the weight matrix
    # Note: In a real run, this is W * M (Effective Weights), but raw W gives us the baseline potential.
    eigenvalues = np.linalg.eigvals(W)
    
    # The Spectral Radius is the absolute max of eigenvalues
    spectral_radius = np.max(np.abs(eigenvalues))
    print(f"Spectral Radius (rho): {spectral_radius:.4f}")
    
    if spectral_radius < 0.9:
        print("-> Regime: SUB-CRITICAL (Signals die out fast. Good for simple tasks.)")
    elif spectral_radius > 1.2:
        print("-> Regime: SUPER-CRITICAL (Signals explode. Risk of chaos.)")
    else:
        print("-> Regime: CRITICAL / EDGE OF CHAOS (Ideal for complex memory.)")

    # --- 4. VISUALIZATION ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Weight Histogram
    ax1.hist(W.flatten(), bins=100, color='purple', alpha=0.7, log=True)
    ax1.set_title("Excitatory/Inhibitory Balance (Weight Dist)")
    ax1.set_xlabel("Weight Value")
    ax1.set_ylabel("Count (Log Scale)")
    ax1.axvline(0, color='k', linestyle='--', linewidth=1)
    ax1.text(0.05, 0.9, "Excitatory >", transform=ax1.transAxes, color='green')
    ax1.text(0.7, 0.9, "< Inhibitory", transform=ax1.transAxes, color='red')

    # Plot 2: Eigenvalue Spectrum (The "Echo" Circle)
    # Real part vs Imaginary part
    ax2.scatter(eigenvalues.real, eigenvalues.imag, alpha=0.1, s=2)
    ax2.set_title(f"Eigenvalue Spectrum (Radius = {spectral_radius:.2f})")
    ax2.set_xlabel("Real Axis")
    ax2.set_ylabel("Imaginary Axis")
    
    # Draw Unit Circle
    circle = plt.Circle((0, 0), 1.0, color='r', fill=False, linestyle='--', linewidth=2, label='Unit Circle (Stability)')
    ax2.add_patch(circle)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Ensure you point this to your actual .pt file if it exists
    analyze_brain_physics("logs/hopfield_rl_model.pt")