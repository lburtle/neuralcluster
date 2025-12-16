
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
    
    save_path = "mnist_evaluation_results.png"
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    evaluate_mnist_capabilities()
