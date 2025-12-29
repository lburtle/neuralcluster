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
# MNIST Task (0-784) + RL Task (800-1611: 27 Ant + 784 MNIST)
# Note: These are used if use_states=True to clamp specific neurons
mnist_range = list(range(784))
rl_range = list(range(800, 800 + 811))
input_neurons = sorted(list(set(mnist_range + rl_range))) 
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
# Topology-Aware Layout
from models.integrated_hopfield_rl import V1_END, V2_END, VENTRAL_END, DORSAL_END

def get_topology_pos(G):
    pos = {}
    import math
    for node in G.nodes():
        # Add random jitter to avoid perfect overlap
        x_jitter = np.random.uniform(-0.5, 0.5)
        y_jitter = np.random.uniform(-0.5, 0.5)
        z_jitter = np.random.uniform(-0.2, 0.2)
        
        if node < V1_END: # V1 (Bottom)
            # Grid layout for V1
            row = (node % 22) 
            col = (node // 22)
            pos[node] = np.array([col + x_jitter, row + y_jitter, 0 + z_jitter])
        elif node < V2_END: # V2 (Middle)
            idx = node - V1_END
            row = (idx % 20)
            col = (idx // 20)
            pos[node] = np.array([col + x_jitter, row + y_jitter, 15 + z_jitter])
        elif node < VENTRAL_END: # Ventral (Top Left)
            idx = node - V2_END
            row = (idx % 17)
            col = (idx // 17)
            # Shift Left
            pos[node] = np.array([col - 20 + x_jitter, row + y_jitter, 30 + z_jitter])
        else: # Dorsal (Top Right)
            idx = node - VENTRAL_END
            row = (idx % 36) # Larger layer
            col = (idx // 36)
            # Shift Right
            pos[node] = np.array([col + 20 + x_jitter, row + y_jitter, 30 + z_jitter])
    return pos

def visualize_3d(G, partition):
    # Hybrid Layout: Initialize with Topology, then Relax with Springs
    print("Computing Topology-Aware Initialization...")
    init_pos = get_topology_pos(G)
    
    # Scale initial positions to be compatible with spring layout's expected 0-1 range (optional, but good practice)
    # Actually spring_layout centers things. Let's just pass it.
    # We use a lower number of iterations to keep it "Anchored" to the structure
    # 'k' is the optimal distance between nodes. 
    # For 2500 nodes, we want small k. 1/sqrt(N) is default ~ 0.02.
    print("Refining layout with Force-Directed Physics (Hybrid)...")
    pos_3d = nx.spring_layout(G, dim=3, pos=init_pos, iterations=50, seed=42, scale=100.0)
    
    # Organize edges by weight strength for visualization
    strong_edges_x, strong_edges_y, strong_edges_z = [], [], []
    medium_edges_x, medium_edges_y, medium_edges_z = [], [], []
    weak_edges_x, weak_edges_y, weak_edges_z = [], [], []
    
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_w = max(weights) if weights else 1.0
    min_w = min(weights) if weights else 0.0
    
    # Simple binning logic
    threshold_strong = min_w + 0.8 * (max_w - min_w)
    threshold_medium = min_w + 0.5 * (max_w - min_w)
    
    for edge in G.edges(data=True):
        x0, y0, z0 = pos_3d[edge[0]]
        x1, y1, z1 = pos_3d[edge[1]]
        w = edge[2]['weight']
        
        if w >= threshold_strong:
            strong_edges_x += [x0, x1, None]
            strong_edges_y += [y0, y1, None]
            strong_edges_z += [z0, z1, None]
        elif w >= threshold_medium:
            medium_edges_x += [x0, x1, None]
            medium_edges_y += [y0, y1, None]
            medium_edges_z += [z0, z1, None]
        else:
            weak_edges_x += [x0, x1, None]
            weak_edges_y += [y0, y1, None]
            weak_edges_z += [z0, z1, None]
    
    # Create three separate traces for edges
    # Increase opacity/width to ensure they are seen
    trace_weak = go.Scatter3d(x=weak_edges_x, y=weak_edges_y, z=weak_edges_z, mode='lines', 
                              line=dict(width=1, color='rgba(200, 200, 200, 0.1)'), name='Weak Connections', hoverinfo='none')
    
    trace_medium = go.Scatter3d(x=medium_edges_x, y=medium_edges_y, z=medium_edges_z, mode='lines', 
                                line=dict(width=3, color='rgba(100, 100, 100, 0.4)'), name='Medium Connections', hoverinfo='none')
                                
    trace_strong = go.Scatter3d(x=strong_edges_x, y=strong_edges_y, z=strong_edges_z, mode='lines', 
                                line=dict(width=6, color='rgba(0, 0, 0, 0.8)'), name='Strong Connections', hoverinfo='none')
    
    # Split Nodes into Regional Traces for Legend
    v1_x, v1_y, v1_z = [], [], []
    v2_x, v2_y, v2_z = [], [], []
    ventral_x, ventral_y, ventral_z = [], [], []
    dorsal_x, dorsal_y, dorsal_z = [], [], []
    
    for node in G.nodes():
        x, y, z = pos_3d[node]
        if node < V1_END:
            v1_x.append(x); v1_y.append(y); v1_z.append(z)
        elif node < V2_END:
            v2_x.append(x); v2_y.append(y); v2_z.append(z)
        elif node < VENTRAL_END:
            ventral_x.append(x); ventral_y.append(y); ventral_z.append(z)
        else:
            dorsal_x.append(x); dorsal_y.append(y); dorsal_z.append(z)

    # Node Traces
    trace_v1 = go.Scatter3d(x=v1_x, y=v1_y, z=v1_z, mode='markers', marker=dict(size=4, color='blue', opacity=0.8), name='V1 (Input/Visual)')
    trace_v2 = go.Scatter3d(x=v2_x, y=v2_y, z=v2_z, mode='markers', marker=dict(size=4, color='green', opacity=0.8), name='V2 (Associative)')
    trace_ventral = go.Scatter3d(x=ventral_x, y=ventral_y, z=ventral_z, mode='markers', marker=dict(size=4, color='red', opacity=0.8), name='Ventral (Memory/MNIST)')
    trace_dorsal = go.Scatter3d(x=dorsal_x, y=dorsal_y, z=dorsal_z, mode='markers', marker=dict(size=4, color='orange', opacity=0.8), name='Dorsal (Motor/RL)')
    
    layout = go.Layout(
        title='3D Visual-Motor Brain (Hybrid Force-Directed)', 
        width=1200,
        height=1000,
        scene=dict(
            xaxis=dict(title='X'), 
            yaxis=dict(title='Y'), 
            zaxis=dict(title='Z'), 
            aspectmode='data'
        ),
        showlegend=True,
        legend=dict(x=0, y=1)
    )
    
    fig = go.Figure(data=[trace_weak, trace_medium, trace_strong, trace_v1, trace_v2, trace_ventral, trace_dorsal], layout=layout)
    output_file = "energy_graph.html"
    fig.write_html(output_file)
    print(f"Saved Hybrid 3D graph with Legends to {output_file}")

# Run clustering
batch, _ = next(iter(train_loader))
# --- 5. Move clustering batch to device ---
# This is important if you use use_states=True
batch = batch.flatten(1).to(device)

best_k = find_best_k(model, min_k=10000, max_k=50000, step=2000, max_components=50, input_neurons=input_neurons, input_values=batch, use_states=False)
G = get_energy_graph(model, k=best_k, input_neurons=input_neurons, input_values=batch, use_states=False)
communities = greedy_modularity_communities(G, weight='weight')
partition = {node: i for i, comm in enumerate(communities) for node in comm}
print("Detected clusters:", partition)
cluster_sizes = [len(comm) for comm in communities]
print("Cluster sizes:", cluster_sizes)

# Note: The 3D visualization is very slow for 1000 nodes.
visualize_3d(G, partition)