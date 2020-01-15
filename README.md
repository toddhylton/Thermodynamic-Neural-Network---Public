# Thermodynamic Neural Network

## Introduction
This file describes the software organization of the Thermodynamically Neural Network.  Two version are included in the respository with the distinctions "v18" and "v21" included in the filenames.  A manuscript (describing the model in v21) submitted to the journal Entropy is in the repository.  I welcome your comments and questions.  My email is thylton@ucsd.edu.

## Abstract
A thermodynamically motivated neural network model is described that self-organizes to transport charge associated with internal and external potentials while in contact with a thermal reservoir. The model integrates techniques for rapid, large-scale, reversible, conservative equilibration of node states and slow, small-scale, irreversible, dissipative adaptation of the edge states as a means to create multiscale order.  All interactions in the network are local and the network structures can be generic and recurrent.  Isolated networks show multiscale dynamics, and externally driven networks evolve to efficiently connect external positive and negative potentials. The model integrates concepts of conservation, potentiation, fluctuation, dissipation, adaptation, equilibration and causation to illustrate the thermodynamic evolution of organization in open systems.  A key conclusion of the work is that the transport and dissipation of conserved physical quantities drives the self-organization of open thermodynamic systems.

## Running the simulation
To execute a simulation, edit params.py and at the command prompt enter

```
python network_v21.py
```

## Repository Contents

The code is distributed over five files in the repo – params.py, network.py, nodes.py, synapse.py. render.py

### File **params.py**

Class **Parameters** - defines simulation

 Module *__init__*
* defines the network via selection of network geometry, nodes and synapses
* defines execution parameters
* specifies outputs
* builds data structure to support network build and execution
* creates folder to store results
* instantiated by with a call from class Network


### File **network.py**

Class **Network** - builds and runs the network simulation

Module *__init__*
* creates output files and stores simulation definition parameters
* builds the network graph
* creates node positions and assigns node types
* builds the node objects by invoking class **MakeNode** in file **nodes.py**
* creates synapse positions and assigns synapse types
* builds the synapse objects by invoking class **MakeSynapse** in file **synapse.py**
* verifies node and synapse placements
* connects modules in node and synapse objects so that they can communicate state variables

Module *run_network*
* runs the simulation
* collects and stores statistics
* prints statistics to the terminal as the simulation proceeds

Module *print_network*
* prompts for and prints network, node and synapse state variables to the terminal.  Typically used to debug simulation.

Module *kill_simulation*
* terminates simulation and deletes output files.  Called by *__init__* when errors are detected in the network build

Module *__main__*
* instantiates the network
* calls *run_network*
* calls *print_network*
* calls *display* in **render.py** to display and save videos of the node state evolution
* calls *makeplots* in **render.py** to create plots of network statistics vs time


### File **nodes.py**

Class **Node** - describes internal network nodes

Module *__init__*
* builds node data structures

Module *add_synapse*
* builds data structure to attach a synapse to the node

Module *receive_context*
* captures state input from synapses

Module *sample_state*
* reversibly samples node state given current synapse state variables
* sends node state to synapses

Module *update_state*
* irreversibly samples node state given current synapse state variables
* updates synapse charge states
* computes synapse weight updates
* sends node state and weight updates to synapses

Class **Bias** – describes external bias nodes, inherits from **Node**

Module *set_state*
* chooses node state depending on the execution parameters

Module *update_state*
* refines node state selection depending on node inputs
* sends node state changes and weight updates to synapses

Module *evaluate_state*
* reports node state values


### File **synapse.py**

Class **Synapse** - describes synapse weights, communicates node states

Module *__init__*
* initiates data structures

Module *add_nodes*
* builds data structure to attach nodes to the synapse

Module *push_state*
* transfers node potentials through the synapse without updating synapse weight

Module *update_state*
* transfers node potential through the synapse
* updates synapse weight

Class **Real1** – inherits from **Synapse**

Module *update_weight*
* updates the weight of real valued synapse bounded on an interval
* uses simple Gaussian with cutoff at interval bounds

* typically preferred over Real2

Class **Real2** – inherits from **Synapse**

Module *update_weight*
* updates the weight of real valued synapse bounded on an interval
* uses error function deal with the interval bounds

Class **Fixed** – inherits from **Synapse**

Module *update_weight*
* dummy function for a weight that does not change
* typically used to connect an external bias node to network nodes


### File **render.py**

Module *display*
* formats, displays and stores images and videos of the node states and node state changes

Module *makeplots*
* formats and stores plots of network statistics vs simulation step
