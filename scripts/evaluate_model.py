
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import sys
import os

# Ensure we can import from the models directory
sys.path.append(os.getcwd())
try:
    from models.integrated_hopfield_rl import HopfieldEnergyNet, train_mnist, train_rl, device
except ImportError:
    # If running from PIML/ directly, models is a package
    from models.integrated_hopfield_rl import HopfieldEnergyNet, train_mnist, train_rl, device

def evaluate_mnist_capabilities():
    print("--- 1. Training Model (using current settings) ---")
    # We reuse the training function which returns the trained model
    model = train_mnist()

    print("\n--- 1.5. Training Model on RL (Transfer Learning) ---")
    # Transfer the MNIST-trained model to the RL task and continue training
    model = train_rl(pretrained_net=model)

    print("\n--- 2. Evaluating on Test Set ---")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    model.eval()
    test_loss = 0
    correct = 0
    
    # Indices needed for the model
    input_indices = list(range(784))
    output_indices = list(range(784, 794))
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # Flatten
            data = data.view(data.size(0), -1)
            
            # Forward
            outputs, _ = model(input_indices, data, output_indices=output_indices)
            
            # Sum up batch loss
            test_loss += F.cross_entropy(outputs, target, reduction='sum').item()
            
            # Get the index of the max log-probability
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

    # --- Visualization ---
    print("--- 3. Visualizing Predictions ---")
    data, target = next(iter(test_loader))
    data, target = data.to(device), target.to(device)
    flattened_data = data.view(data.size(0), -1)
    
    with torch.no_grad():
        outputs, _ = model(input_indices, flattened_data, output_indices=output_indices)
        preds = outputs.argmax(dim=1)
    
    # Plot first 5 images
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        ax = axes[i]
        img = data[i].cpu().numpy().reshape(28, 28)
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Pred: {preds[i].item()} | True: {target[i].item()}")
        ax.axis('off')
    
    save_path = "images/mnist_evaluation_results.png"
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")

def run_activation_probe(model, device):
    print("\n--- Diagnostic: Activation Probe (Bridge Verification) ---")
    model.eval()
    
    # Define our functional blocks
    ventral_range = range(900, 1200)
    dorsal_range = range(1200, 1500)
    
    # Load 1 Even (e.g., '2') and 1 Odd (e.g., '7') digit
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform), batch_size=1, shuffle=True)

    samples = {'Even': None, 'Odd': None}
    while samples['Even'] is None or samples['Odd'] is None:
        img, label = next(iter(test_loader))
        if label.item() % 2 == 0 and samples['Even'] is None:
            samples['Even'] = (img.view(1, -1).to(device), label.item())
        elif label.item() % 2 != 0 and samples['Odd'] is None:
            samples['Odd'] = (img.view(1, -1).to(device), label.item())

    for category, (img_data, label) in samples.items():
        # Run Hopfield settling (No Ant input, just Vision)
        # Note: We provide 0s for the Ant-state indices (0-26)
        input_indices = list(range(784))
        with torch.no_grad():
            full_states = model(input_indices, img_data, steps=20)
            
        # Measure Mean Absolute Activation in each block
        ventral_act = full_states[0, ventral_range].abs().mean().item()
        dorsal_act = full_states[0, dorsal_range].abs().mean().item()
        
        print(f"[{category} Digit: {label}]")
        print(f" -> Ventral Block (Identity) Activation: {ventral_act:.4f}")
        print(f" -> Dorsal Block (Motor) Activation:  {dorsal_act:.4f}")

    print("\nInterpretation: If Dorsal Activation > 0, your Bridges are alive.")
    print("If Dorsal Activation for 'Even' differs significantly from 'Odd', the bridge has learned the rule!")

import seaborn as sns

def visualize_bridge_heatmaps(model):
    print("\n--- Generating Bridge Connectivity Heatmaps ---")
    model.eval()
    
    # Define block indices
    # Ventral (What): 1800-2300
    # Dorsal (Action): 2300-3500
    v_start, v_end = 1800, 2300
    d_start, d_end = 2300, 3500
    
    # Extract the bridge weights from the full weight matrix
    # We want weights WHERE Ventral is the input and Dorsal is the output
    with torch.no_grad():
        full_weights = model.weights * model.connectivity_mask
        bridge_weights = full_weights[v_start:v_end, d_start:d_end].cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(bridge_weights, cmap="RdBu_r", center=0)
    plt.title("Functional Bridge: Ventral (Y) to Dorsal (X) Connectivity")
    plt.xlabel("Dorsal Neurons (Motor indices 2300-3500)")
    plt.ylabel("Ventral Neurons (Vision indices 1800-2300)")
    
    save_path = "images/bridge_heatmap.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved bridge connectivity heatmap to {save_path}")

def run_contrast_diagnostic(model, device):
    print("\n--- Diagnostic: State Separation (Euclidean Distance) ---")
    model.eval()

    # 1. Prepare Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform), batch_size=1, shuffle=True)

    even_img, odd_img = None, None
    
    # Iterate until we find one of each to compare
    for img, label in test_loader:
        if label.item() % 2 == 0 and even_img is None:
            even_img = img.view(1, -1).to(device) # Flatten to (1, 784)
            even_label = label.item()
        elif label.item() % 2 != 0 and odd_img is None:
            odd_img = img.view(1, -1).to(device) # Flatten to (1, 784)
            odd_label = label.item()
        
        if even_img is not None and odd_img is not None:
            break

    print(f"Comparing Even Digit ({even_label}) vs Odd Digit ({odd_label})")
    
    # We want to look specifically at the Dorsal (Motor) indices: 1200-1500
    dorsal_indices = torch.arange(1200, 1500).to(device)
    
    with torch.no_grad():
        # Settle the network for both inputs
        even_state = model(list(range(784)), even_img, steps=100)
        odd_state = model(list(range(784)), odd_img, steps=100)
        
        # Isolate the Motor Block's response
        v_even = even_state[0, dorsal_indices]
        v_odd = odd_state[0, dorsal_indices]
        
        # Calculate Euclidean Distance
        dist = torch.norm(v_even - v_odd, p=2).item()
        
        # Calculate "Angle" (Cosine Similarity) 
        # 1.0 = identical patterns, 0.0 = orthogonal/totally different
        cos_sim = F.cosine_similarity(v_even.unsqueeze(0), v_odd.unsqueeze(0)).item()

    print(f"Distance between Even/Odd thoughts: {dist:.4f}")
    print(f"Cosine Similarity (1.0 is bad): {cos_sim:.4f}")
    
    if cos_sim > 0.95:
        print("RESULT: High Similarity. The Motor block is ignoring the 'What' signal.")
    elif cos_sim < 0.70:
        print("RESULT: High Differentiation! The bridge is functioning as a selector.")

def verify_architecture_integrity():
    print("\n" + "="*50)
    print("--- ARCHITECTURE INTEGRITY SANITY CHECK ---")
    print("="*50)
    
    # 1. Check Block Alignment
    blocks = [
        ("V1 (Sensory)", 0, V1_END),
        ("V2 (Associative)", V1_END, V2_END),
        ("Ventral (Vision)", V2_END, VENTRAL_END),
        ("Dorsal (Motor)", VENTRAL_END, DORSAL_END)
    ]
    
    for name, start, end in blocks:
        size = end - start
        print(f"-> {name:18} | Range: {start:4}-{end:4} | Size: {size:4}")
        if size <= 0:
            print(f"   [!] ERROR: Block {name} has zero or negative size!")

    # 2. Check Input Mapping
    ant_dim = 27
    mnist_dim = 784
    total_input = ant_dim + mnist_dim # 811
    
    print(f"\n-> Input Mapping:")
    print(f"   Ant Proprioception Port: Neurons 0 - {ant_dim-1}")
    print(f"   MNIST Vision Port:        Neurons {ant_dim} - {total_input-1}")
    
    if total_input > V1_END:
        print(f"   [!] WARNING: Inputs ({total_input}) exceed V1 capacity ({V1_END})!")
    else:
        print(f"   [v] Success: Inputs fit 1-to-1 inside V1 (Port Margin: {V1_END - total_input})")

    # 3. Check Monitor Ranges
    v_mon = (V2_END, VENTRAL_END)
    d_mon = (VENTRAL_END, DORSAL_END)
    print(f"\n-> Monitor Configuration:")
    print(f"   V-Monitor (Reading 'What'):   {v_mon}")
    print(f"   D-Monitor (Reading 'Action'): {d_mon}")
    
    if v_mon[1] > d_mon[0]:
        print("   [!] ERROR: Overlap detected between Ventral and Dorsal monitors!")
    
    print("="*50 + "\n")

if __name__ == "__main__":
    # 1. This trains/loads the model and checks if it can still do MNIST
    evaluate_mnist_capabilities()
    
    # 2. This is the new diagnostic to see IF the vision is talking to the motor
    # We pass the model that was just evaluated
    from models.integrated_hopfield_rl import device, NUM_NEURONS
    # (Assuming evaluate_mnist_capabilities is modified to return the model, 
    # or you load the saved .pt file)
    
    # Example if you load the trained model:
    model = HopfieldEnergyNet(num_neurons=NUM_NEURONS).to(device)
    model.load_state_dict(torch.load("logs/hopfield_rl_model.pt"))
    
    run_activation_probe(model, device)
    run_contrast_diagnostic(model, device)

    # 3. Generate the visual map (Which signals are crossing?)
    visualize_bridge_heatmaps(model)
