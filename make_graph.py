import numpy as np
import networkx as nx
from networkx.algorithms.community import modularity, greedy_modularity_communities
from kneed import KneeLocator
import plotly.graph_objects as go
import torch
from torchvision import datasets, transforms
import sys
import os

# Ensure we can import from the models directory
sys.path.append(os.getcwd())
try:
    from models.integrated_hopfield_rl import HopfieldEnergyNet, train_mnist, device, NUM_NEURONS, DEFAULT_LOG_DIR
except ImportError:
    # If running from PIML/ directly, models is a package
    from models.integrated_hopfield_rl import HopfieldEnergyNet, train_mnist, device, NUM_NEURONS, DEFAULT_LOG_DIR

# Configuration
n_input = 784 # MNIST input size

# Load or Train Model
# Since we don't have a saved file, we'll train one quickly or use the function
# Load Model
model_path = os.path.join(DEFAULT_LOG_DIR, "hopfield_rl_model.pt")
model = HopfieldEnergyNet(num_neurons=NUM_NEURONS).to(device)

if os.path.exists(model_path):
    print(f"Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print("Model file not found. Please run 'python models/integrated_hopfield_rl.py --mode rl' first to generate weights.")
    # Fallback to fresh initialization (random weights) if we just want to see the graph structure, 
    # but ideally we want trained weights.
    print("Using initialized random weights.")

# If you want to save it: torch.save(model.state_dict(), "mnist_model.pt")

## Should def switch to loading from file in the future, retraining might be a problem
## Also should train model simultaneously for both tasks

model.eval()

# Define input range for graph analysis
# Define input range for graph analysis
# MNIST (0-783) + Ant (794-820, assuming 27 inputs)
ant_obs_size = 27
input_neurons = list(range(n_input)) + list(range(794, 794 + ant_obs_size)) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup Data Loader for batch processing
print("Setting up data loader...")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# Use a small subset for graph generation if needed, but train_loader is standard
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)

# Optimized find_best_k
def get_energy_graph(model, k=3000, input_neurons=None, input_values=None, use_states=False):
    if use_states and input_neurons is not None and input_values is not None:
        with torch.no_grad():
            # input_values should be on the same device as model
            # output_neurons for MNIST are 784-794
            output_neurons = list(range(784, 794))
            _, states = model(input_neurons, input_values, output_indices=output_neurons)
            states = states.cpu().numpy() # Move to CPU for numpy
            weight_matrix = np.corrcoef(states.T)
            weight_matrix = np.nan_to_num(weight_matrix, 0)
    else:
        # Move weights to CPU for numpy
        weight_matrix = model.weights.data.cpu().numpy()
        weight_matrix = (weight_matrix - weight_matrix.min()) / (weight_matrix.max() - weight_matrix.min() + 1e-8)
        
    print(f"Weight matrix stats: min={weight_matrix.min():.4f}, max={weight_matrix.max():.4f}, std={np.std(weight_matrix):.4f}")
    G = nx.Graph()
    edges = []
    for i in range(model.num_neurons):
        for j in range(i + 1, model.num_neurons):
            edges.append((i, j, weight_matrix[i, j]))
            
    edges = sorted(edges, key=lambda x: x[2], reverse=True)[:k]
    for i in range(model.num_neurons):
        G.add_node(i)
    for i, j, w in edges:
        G.add_edge(i, j, weight=w)
        
    singletons = sum(1 for node in G.nodes() if G.degree(node) == 0)
    print(f"Singletons: {singletons}/{model.num_neurons}")
    return G

def find_best_k(model, min_k=3000, max_k=15000, step=1000, max_components=20, input_neurons=None, input_values=None, use_states=False):
    k_values = []
    mod_values = []
    for k in range(min_k, max_k + 1, step):
        G = get_energy_graph(model, k=k, input_neurons=input_neurons, input_values=input_values, use_states=use_states)
        components = nx.number_connected_components(G)
        communities = greedy_modularity_communities(G, weight='weight')
        current_mod = modularity(G, communities, weight='weight')
        if components <= max_components:
            k_values.append(k)
            mod_values.append(current_mod)
            cluster_sizes = [len(comm) for comm in communities]
            print(f"k={k}: Modularity={current_mod:.4f}, Components={components}, Cluster sizes={cluster_sizes}")
    
    if not k_values:
        print("No valid k found; try increasing max_components or min_k")
        return min_k
    
    kneedle = KneeLocator(k_values, mod_values, curve="convex", direction="decreasing")
    best_k = kneedle.knee if kneedle.knee else k_values[np.argmax(mod_values)]
    print(f"Best balanced k: {best_k} (modularity={max(mod_values):.4f})")
    return best_k

# Visualization
def visualize_3d(G, partition):
    pos_3d = nx.spring_layout(G, dim=3, seed=42)
    edge_x, edge_y, edge_z = [], [], []
    for edge in G.edges():
        x0, y0, z0 = pos_3d[edge[0]]
        x1, y1, z1 = pos_3d[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]
    edge_trace = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(width=2, color='gray'))
    
    node_x, node_y, node_z = [], [], []
    node_colors = []
    colors = {0: 'blue', 1: 'red', 2: 'green', 3: 'purple', 4: 'orange', 5: 'cyan', 6: 'yellow', 7: 'pink', 8: 'brown', 9: 'black'}
    for node in G.nodes():
        x, y, z = pos_3d[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_colors.append(colors.get(partition.get(node, 0), 'gray'))
        
    node_trace = go.Scatter3d(x=node_x, y=node_y, z=node_z, mode='markers+text', marker=dict(size=10, color=node_colors), text=list(G.nodes()))
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        title='3D Emergent Neuron Clusters', 
        width=800,
        height=800,
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='cube')
    ))
    # fig.show()
    output_file = "energy_graph.html"
    fig.write_html(output_file)
    print(f"Saved 3D graph to {output_file}")

# Run clustering
batch, _ = next(iter(train_loader))
# --- 5. Move clustering batch to device ---
# This is important if you use use_states=True
batch = batch.flatten(1).to(device)

best_k = find_best_k(model, min_k=3000, max_k=15000, step=1000, max_components=20, input_neurons=input_neurons, input_values=batch, use_states=False)
G = get_energy_graph(model, k=best_k, input_neurons=input_neurons, input_values=batch, use_states=False)
communities = greedy_modularity_communities(G, weight='weight')
partition = {node: i for i, comm in enumerate(communities) for node in comm}
print("Detected clusters:", partition)
cluster_sizes = [len(comm) for comm in communities]
print("Cluster sizes:", cluster_sizes)

# Note: The 3D visualization is very slow for 1000 nodes.
visualize_3d(G, partition)