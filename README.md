# NeuralCluster

**NeuralCluster** is a research repository investigating **emergent behavior** and **functional clustering** within energy-based neural networks (specifically Hopfield-style networks). 

The project explores how distinct functional regions such as sensory input (Vision) and motor control (RL) can be integrated into a single, unstructured neural fabric, allowing "bridges" and specialized clusters to form naturally through training.

---

## Core Concepts

### 1. Energy-Based Learning
Unlike traditional feed-forward networks, models in this repository (like `HopfieldEnergyNet`) rely on **neuron injection** and **energy minimization**. The network settles into a stable low-energy state that represents the solution or classification.

The neuron injection looks like ```xor_inputs = torch.tensor([[-1.,-1.], [-1.,1.], [1.,-1.], [1.,1.]], dtype=torch.float32)``` with the neuron initialized as ```xor_in_neurons = [0, 1]```. We can also specify what neurons we monitor for outputs: ```xor_out_neurons = [2]```, and focus the training for each task simultaneously, concurrently monitoring several different task losses. From this, I looked towards more complex tasks being contained within the same network, since we can differentiate the output neurons and input neurons, only sharing functional pathways through the network.

### 2. Neuromorphic Architecture
To achieve a more complex architecture, I implemented a biologically inspired design which allowed for more discete information flow. This was not to explicitly form clusters in the network, but instead to prevent the inputs from the RL environment from being crossed with input from the MNIST digits, though this is later joined. The network is conceptually divided into regions to mimic biological structures, though connections can form globally:
- **V1 (Sensory)**: Input handling.
- **V2 (Associative)**: Intermediate processing.
- **Ventral Stream**: The "What" pathway (e.g., MNIST digit recognition).
- **Dorsal Stream**: The "Where/Action" pathway (e.g., RL/Motor control).

The ***Sensory block (V1)*** is where the 28x28 MNIST images were ingested. From there, it is immediately connected to the ***Associative block (V2)*** which handles forming higher order pattern recognition ***without*** Convolutional layers. This is in essence just an MLP, but with directed flow through an energy network. The number patterns are then passed to the ***Ventral Stream*** where they are used for final digit classification. 

On the other side, the ***Dorsal Stream*** handles the entirety of the RL task, where the motor and positional feed are the inputs and the motor control is the output. The important feature is the bridge between the **Dorsal** and **Ventral** which act as a pseudo long-range connection that allows communication between different functional areas. In this case, the MNIST images were incorporated into the RL environment, and whether the digit was **Odd** or **Even** would dictate the reward for moving, motivating to prevent movement during one case. 

### 3. Emergent Clustering
We analyze the network to observe how neurons group together based on connection strength. This is visualized using graph theory techniques to show how the "Vision" part of the brain talks to the "Motor" part. Multiple different clustering algorithms were used, one such being the Greedy Community, to see which neurons shared the strongest weight connections. Hopefully, from the training, the clusters of related neurons (e.g. visual input, ventral stream).

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

## Visualizations

### Network Topology & Clustering
*Visualizing the formation of clusters based on connection strength.*

![3D Optimized Network](images/3D-optimizedk.png)

### Task-Specific Networks
**4-Task Logic Network**

![](images/4task30.png)

**MNIST Classification Network**

![](images/3D_mnist.png)

**Integrated MNIST + RL Network**

![](images/3D_MNIST_RL.png)

---

## Getting Started

### Packages
- Python 3.8+
- PyTorch
- NetworkX
- Plotly
- Matplotlib
- Kneed (Knee detection for clustering)

### Running the Analysis

**1. Train model & Evaluate Performance**
```bash
python scripts/evaluate_model.py
```
*Output: MNIST accuracy and Bridge Activation heatmaps.*
***Run this to train on both tasks and evaluate***.

**2. Check Network Physics (Stability)**
```bash
python scripts/EI.py
```
*Output: Spectral radius and E/I balance plots.*

**3. Visualizing the Brain Graph**
```bash
python scripts/make_graph.py
```
*Output: Generates a 3D HTML graph of the network.*


---

## Future Directions
- **Attention-Based Clustering**: Further exploration of global attention mechanisms (Transformer-style) vs local energy dynamics.
- **Scalability**: Scaling the neuron count while maintaining the "Critical" stability regime.
