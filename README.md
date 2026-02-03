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

On the other side, the ***Dorsal Stream*** handles the entirety of the RL task, where the motor and positional feed are the inputs and the motor control is the output. The important feature is the bridge between the **Dorsal** and **Ventral** which act as a pseudo long-range connection that allows communication between different functional areas. In this case, the MNIST images were incorporated into the RL environment, and whether the digit was **Odd** or **Even** would dictate the reward for moving, motivating to prevent movement during one case. The integrated environment is a total of 784 + 27 input dimensions, with 13 positional values and 14 velocity values in those 27. The environment is crudely visualized in the generated video, looking as follows:

<img src="images/example_rl_env.PNG" width="50%"/>

### 3. Emergent Clustering
We analyze the network to observe how neurons group together based on connection strength. This is visualized using graph theory techniques to show how the "Vision" part of the brain talks to the "Motor" part. Multiple different clustering algorithms were used, one such being the Greedy Community, to see which neurons shared the strongest weight connections. Hopefully, from the training, the clusters of related neurons (e.g. visual input, ventral stream).

---

## Repository Structure

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

### Task-Specific Networks
**4-Task Logic Network**

2D connection visualization

(Flat clustered projection of the 4 task network)

<img src="images/4task30.png" width="50%"/>

3D connection visualization

(3D clustered projection of the 4 task network. Shows the depth of the clusters that was not apparent in 2 dimensions. No overlapping clusters. I recommend running the code itself for the interactive version)

<img src="images/3D-optimizedk.png" width="50%"/>

**MNIST Classification Network**

(Clustered 3D visualization of the MNIST only trained network. Showed clear separating between clusters. Unsure if they were functional groups. Still just a semi-yarn ball of neurons.)

<img src="images/3D_mnist.png" width="50%"/>

**Integrated MNIST + RL Network**

(Clustered 3D visualization of the network trained on MNIST and RL tasks. Showed more pronounced subgroups within the network. No longer just a yarn ball, but instead had physically separated groups.)

<img src="images/3D_MNIST_RL.png" width="50%"/>

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
## Results

When training the network on both the MNIST recognition task and RL motor task, we see a degradation in the performance on the MNIST set. On my most recent run, the model was trained for on 10,000 MNIST images in batches of 64 and 200,000 steps on the RL task. A test accuracy of **10.53%** is observed (at least in my most recent run) meaning that it is essentially guessing randomly. The model **technically** correctly doesn't move on an odd digit, but over mutliple runs, it is obvious that it either fails to learn the odd/even rule, or just fails to learn proper locomotion. We can visualize the test accuracy of the MNIST classification:

<img src="images/mnist_evaluation_results.png" width="75%"/>

However, more was monitored in this network than simply the performance. The goal was not simply to see if the network would learn, but if it could build functional connections between the tasks. This was measured with a few different metrics. First, the cosine similarity was measured between the states of the network when presented with an even or odd digit. This was to measure how dissimilar the "thoughts" or saved states were for each input. It acheived a **0.5419**, meaning that the two vectors were fairly similar, with a 1.0 meaning total similarity, and -1.0 being completely different. The Euclidean Distance, or L2 Norm, was measured between these two state vectors as well to determine if the network had separated the even and odd states into distinct classes in the latent motor space. The model achieved a distance of **8.3904**. 

I also monitored the connection strength of the neurons specifically contained within the bridge between the Ventral and Dorsal blocks. If it had a value of 0, that meant that there was no connection between the two, meaning there is no cross talk between functional regions. Conversely, a higher value would proportionately indicate the strength of connection between the two. The goal was to allow for these connections to emergently develop, allowing the model to figure out their importance if any, in an integrated task. These activation values are the mean values of all neurons in their blocks. 

On the digit 2, I measured a Ventral Block activation of **0.5462** and Dorsal Block activation of **0.6129**. 
On the digit 7, I measured a Ventral Block activation of **0.4165** and Dorsal Block activation of **0.4772**. 
The important takeaway is the similarity in the values. Since they are fairly or very similar, this means that the bridge did not learn the even/odd rule, but it still maintained a connection. We can get a better understanding of what this means by looking at a heatmap of the bridge connections: 

<img src="images/bridge_heatmap.png" width="50%"/>

It is clear that, while the connections exist, there are not strong patterns or relationships between neuron groups. If there were, we might see something more stratified patterns in the heatmap. Another, less important result to note is the Synaptic Plasticity heatmap, showing which neurons are changing within the bridge at certain intervals. 

<img src="images/bridge_diff.png" width="50%"/>

It may look very similar to the previous heatmap, and that is because they do look similar. But that is because they are essentially just noise. This diff map shows that there is essentially random small adjustments being made towards the end of the training, showing no real signal or direction. The weights are likely randomly adjusting throughout training to try and find some adequate configuration, but the network fails to actually grasp the true relationship, so the connections stay random.

Finally, on a more neuroscience side, I also visualized the Excitatory/Inhibitory (E-I) balance amongst the weights. This reflects the balance between positive and negative connections between neurons, to ensure that the network remains at the regime critical point without experiencing overexcitation or overinhibition. This would look like neurons amplifying signals into white noise or killing signals as they pass through. This is less relevant for Hopfield networks, however, as they do not experience the same dynamics as brains. But it is still helpful to ensure that the network remains at the computational critical point. I also have the Eigenvalue Spectrum of the weights to visualize the eigenvalues of memories encoded in the weight matrix. Greater values equate to stronger memories, or in the context of Hopfield Networks, deeper attractor basins in the energy landscape. The Unit Circle boundary shows the limit to the signal size of the memories, where they are then bound by tanh to prevent from the signal exploding.

<img src="images/EIgraph.png" width="75%"/>

---
## Why this doesn't work

Unfortunately, through enough trial and error, I realized that this method would not work. While there are definitely improvements I could have made in the training process and in other aspects, the main issue lies in the fundamental behavior of Hopfield Networks. With the tasks that I gave the model to learn, I explicitly chose them to be significantly different, yet also functions that are necessary for any real, complex living creature to be able to execute, being vision and motor control. In this case, motor control was specifically controlled and purposeful movement, not just random gyration. 

The most critical feature of Hopfield Networks, which explains why they are able to succeed in image recognition (MNIST) yet fail with RL tasks (movement) is that they are **associative memory machines**. When training a Hopfield Network, we learn the weights via the Hebbian Learning Rule:

<p align="center">
$w_{ij} = \frac{1}{N} \sum_{\mu=1}^{M} \xi_i^\mu \xi_j^\mu$
</p>

Where $`\xi_i^\mu`$ is the state of the i-th neuron for the $`\mu`$-th pattern, likewise for the j-th neuron, and $`\frac{1}{N}`$ is a normalization term. With this, patterns are sculpted into the weight matrix, creating attractor basins in the energy landscape. Building off of this, usually it is better suited for learning fewer, more distinct patterns, since more orthogonal patterns are better stored together in the same matrix, otherwise their attractor basins overlap. However, we still observe that the MNIST digits were distinct enough when flattened to be aptly recognized and classified. 

But this only holds up well for pattern recognition. We see a different side when we introduce RL. When constantly trying to reinforce the network towards **patterns of behavior**, the network is constantly being told to rewire for each instance, where the inputs that it is trying to learn are all nearly identical, but one combination may be positive and another negative. When trying to finely teach an associative memory machine such a fine task, it is akin to just constantly carving through energy landscape and making one massive basin. The model will have no signal of when to move or what to do. This exact behavior is observed when we evaluate the model. Even if we separated the two tasks, it still either moves sporadically, or chooses to stay still no matter what, showing that it cannot distinguish any decision boundaries, or between two attractor basins. 

### Possible things to add
- EWC implementation explanation
- Distance Penalty (relates back to emphasis on localized clusters)
