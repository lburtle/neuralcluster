# NeuralCluster

**NeuralCluster** is a research repository investigating **emergent behavior** and **functional clustering** within energy-based neural networks (specifically Hopfield-style networks). 

The project explores how distinct functional regions—such as sensory input (Vision) and motor control (RL)—can be integrated into a single, unstructured neural fabric, allowing "bridges" and specialized clusters to form naturally through training.

---

## 🧠 Core Concepts

### 1. Energy-Based Learning
Unlike traditional feed-forward networks, models in this repository (like `HopfieldEnergyNet`) rely on **neuron injection** and **energy minimization**. The network settles into a stable low-energy state that represents the solution or classification.

### 2. Neuromorphic Architecture
The network is conceptually divided into regions to mimic biological structures, though connections can form globally:
- **V1 (Sensory)**: Input handling.
- **V2 (Associative)**: Intermediate processing.
- **Ventral Stream**: The "What" pathway (e.g., MNIST digit recognition).
- **Dorsal Stream**: The "Where/Action" pathway (e.g., RL/Motor control).

### 3. Emergent Clustering
We analyze the network to observe how neurons group together based on connection strength. This is visualized using graph theory techniques to show how the "Vision" part of the brain talks to the "Motor" part.

---

## 📂 Repository Structure

### `models/`
Contains the core neural network implementations and training notebooks.
- **`integrated_hopfield_rl.py`**: The primary model definition and training script. Integrates MNIST classification with Reinforcement Learning tasks.
- **`transformernet.ipynb`**: Exploration of attention-based architectures.
- **`4taskhopfield.ipynb`**: Earlier experiments with multi-task logic gates (XOR/AND).

### `scripts/`
Analysis and visualization tools to inspect the "brain" of the network.
- **`make_graph.py`**: Generates **3D interactive graphs** of the neural topology, visualizing clusters and connection weights using NetworkX and Plotly.
- **`EI.py`**: Analyzes the **Physics of the Network**. Checks Excitatory/Inhibitory balance (E/I) and calculates the **Spectral Radius** to ensure the network operates at the "Edge of Chaos" (Critical regime).
- **`evaluate_model.py`**: Runs diagnostics to verify if specific functional bridges (e.g., Vision sees a '2' -> Motor plans 'Action A') are active.

---

## 📊 Visualizations

### Network Topology & Clustering
*Visualizing the formation of clusters based on connection strength.*

![3D Optimized Network](images/3D-optimizedk.png)

### Task-Specific Networks
**4-Task Logic Network**
![4 task network](images/4task30.png)

**MNIST Classification Network**
![MNIST network](images/3D_mnist.png)

**Integrated MNIST + RL Network**
![MNIST + RL](images/3D_MNIST_RL.png)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- NetworkX
- Plotly
- Matplotlib
- Kneed (Knee detection for clustering)

### Running the Analysis

**1. Check Network Physics (Stability)**
```bash
python scripts/EI.py
```
*Output: Spectral radius and E/I balance plots.*

**2. Visualizing the Brain Graph**
```bash
python scripts/make_graph.py
```
*Output: Generates a 3D HTML graph of the network.*

**3. Evaluate Performance**
```bash
python scripts/evaluate_model.py
```
*Output: MNIST accuracy and Bridge Activation heatmaps.*

---

## 🔮 Future Directions
- **Attention-Based Clustering**: Further exploration of global attention mechanisms (Transformer-style) vs local energy dynamics.
- **Scalability**: Scaling the neuron count while maintaining the "Critical" stability regime.
