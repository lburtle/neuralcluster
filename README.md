# neuralcluster
 -------------------------------
## Two Models:
#### - Attention Based Neural Cluster
#### - Energy Based Neural Cluster
## Task
#### Trained on XOR and AND truth tables (0, 1, 1, 0), (0, 0, 0, 1)
#### Given initial inputs [[-1.,-1.], [-1.,1.], [1.,-1.], [1.,1.]] in the form of a tensor

## Training Method
#### Since there is no clear input and output layer, this model instead relies on neuron injection
#### Looks like: input_neurons = [3, 7] output_neuron = [5]
#### This trains a specific pathway through the network to optimize the task
#### Theoretically allows for different tasks to be encoded into different clusters

## Emergent Behavior
#### The goal is to observe distinct clusters forming within the network
#### These clusters are formed based on how strong the connections are between them
#### Visualized using a clustering algorithm which determines how many edges/connections to consider in the projection and assigning each neuron to a cluster

## Network Types Explained
### Attention Based:
#### This uses attention between every neuron in the network to allow for connections with no structure (specifically no layers)
#### Connections are updated based on howw strongly each neuron attends to each other for the given task
#### Connections are not localized, but instead global, allowing connections over any distance

### Energy Based
#### Hopfield Energy Network
#### Have to review
#### Desired structure where neurons can build connections across entire network, but more local ones are preferred
#### Entropy used as a means of loss to optimize the model
#### Network settles to lowest/most stable energy state
#### To be continued
