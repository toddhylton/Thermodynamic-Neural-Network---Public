# Thermodynamic Neural Network Description

import sys
import math
import time as tm
import os

class Parameters(object):
    def __init__(self):

# filename qualifier suffix
        qualifier_string = ''

# simulation output parameters
        self.print_records = 0
        self.show_video = False
        self.save_state_video = True
        self.save_change_video = True
        self.save_images = 0
        self.save_plots = True
        self.delete_state_file = True
        self.delete_plot_file = True

# network type list and component class dictionaries
        self.network_type_list = ['neighbor', 'random', 'gaussian', 'exponential'] 
        self.node_class_list_dict = {}
        self.node_class_list_dict['ordered'] = ['bias', 'discrete']
        self.node_class_list_dict['network'] = ['discrete']
        self.node_class_list_dict['logic'] = ['bias']
        self.node_class_list_dict['compound'] = ['network', 'logic']
        self.node_label_dict = {'network':'Network', 'logic':'Logic'}

# network execution parameters
        self.time = 10
        self.era = {}
        self.era[0]  =  {'epochs': 0,  'weight_update': True, 'logic_mode': 'off'}
        self.era[1]  =  {'epochs': 50,   'weight_update': True, 'logic_mode': 'driven'}
        self.era[2]  =  {'epochs': 0,  'weight_update': True, 'logic_mode': 'off'}
        self.era[3]  =  {'epochs': 0,   'weight_update': True, 'logic_mode': 'driven'}
        self.era[4]  =  {'epochs': 0,  'weight_update': True, 'logic_mode': 'off'}
        self.era[5]  =  {'epochs': 0,   'weight_update': True, 'logic_mode': 'driven'}
        self.era[6]  =  {'epochs': 0,  'weight_update': True, 'logic_mode': 'off'}
        self.era[7]  =  {'epochs': 0,   'weight_update': True, 'logic_mode': 'driven'}
        self.era[8]  =  {'epochs': 0,  'weight_update': True, 'logic_mode': 'off'}
        self.epochs = sum([self.era[i]['epochs'] for i in self.era])

# network architecture parameters
        self.dimension = 2
        self.network = 'neighbor'
        self.scale = 1.0
        self.bipartite = True
        self.bias_node_placement_separation = 10   # 8 for 16 in 1600 /  24 for 8 in 10000 for 2D neighbor networks with 4 connections (or randomly connected networks) / 24 for 32 in 40k 2D neighbor networks / 16 for 32 in 10k neighbor networks
        self.bias_node_link_separation = 5   # 5 for 16 in 1600 for 2D neighbor networks with 4 connections (or randomly connected networks)

# node description parameters
        self.node_dict = {class1: {} for class1 in self.node_class_list_dict['ordered']}
        
        # network node classes
        self.node_dict['discrete'][0]   =   {'quantity': 000,     'states': 2,    'connections': 8,     'recur': 0}
        self.node_dict['discrete'][1]   =   {'quantity': 000,     'states': 3,    'connections': 4,     'recur': 0}
        self.node_dict['discrete'][2]   =   {'quantity': 900,     'states': 10,   'connections': 16,     'recur': 0}

        # logic node classes
        bias_target = 80.0

        self.node_dict['bias'][0] =         {'quantity': 0,     'complement':True,     'connections': 4,    'period': [132],   'part': 'even'}
        self.node_dict['bias'][1] =         {'quantity': 0,     'complement':True,     'connections': 4,    'period': [70],  'part': 'odd'}
        self.node_dict['bias'][2] =         {'quantity': 0,     'complement':True,     'connections': 4,    'period': [211],  'part': 'even'}
        self.node_dict['bias'][3] =         {'quantity': 0,     'complement':True,     'connections': 4,    'period': [105],  'part': 'odd'}

        self.node_dict['bias'][4] =         {'quantity': 0,     'complement':True,     'connections': 4,    'period': [173],  'part': 'even'}
        self.node_dict['bias'][5] =         {'quantity': 0,     'complement':True,     'connections': 4,    'period': [98],  'part': 'odd'}
        self.node_dict['bias'][6] =         {'quantity': 0,     'complement':True,     'connections': 4,    'period': [164],  'part': 'even'}
        self.node_dict['bias'][7] =         {'quantity': 0,     'complement':True,     'connections': 4,    'period': [121],  'part': 'odd'}

        self.node_dict['bias'][8] =         {'quantity': 0,     'complement':True,     'connections': 4,    'period': [222],  'part': 'even'}
        self.node_dict['bias'][9] =         {'quantity': 0,     'complement':True,     'connections': 4,    'period': [150],  'part': 'odd'}
        self.node_dict['bias'][10] =        {'quantity': 0,     'complement':True,     'connections': 4,    'period': [33],  'part': 'even'}
        self.node_dict['bias'][11] =        {'quantity': 0,     'complement':True,     'connections': 4,    'period': [40],  'part': 'odd'}

        self.node_dict['bias'][12] =        {'quantity': 0,     'complement':True,     'connections': 4,    'period': [16],  'part': 'even'}
        self.node_dict['bias'][13] =        {'quantity': 0,     'complement':True,     'connections': 4,    'period': [31],  'part': 'odd'}
        self.node_dict['bias'][14] =        {'quantity': 0,     'complement':True,     'connections': 4,    'period': [37],  'part': 'even'}
        self.node_dict['bias'][15] =        {'quantity': 0,     'complement':True,     'connections': 4,    'period': [43],  'part': 'odd'}
          

# synapse description parameters
        self.synapse_dict = {class1: {class2: {} for class2 in self.node_class_list_dict['ordered']} for class1 in self.node_class_list_dict['ordered']}

        # set connection specifics for allowable connections       
        self.synapse_dict['discrete']['discrete'] =     {'connect': True,    'type': 'real1',   'synapse_ef':80}        
        self.synapse_dict['discrete']['bias'] =         {'connect': True,    'type': 'fixed'}
        self.synapse_dict['bias']['discrete'] =         {'connect': True,    'type': 'fixed'}


# set node and synapse defaults if not specified
        # set node defaults if not specified in node dictionary
        for node_class in self.node_dict:
            for m in self.node_dict[node_class]:
                if 'quantity' not in self.node_dict[node_class][m]: self.node_dict[node_class][m]['quantity'] = 0
                if 'complement' not in self.node_dict[node_class][m]: self.node_dict[node_class][m]['complement'] = False
                if 'states' not in self.node_dict[node_class][m]: self.node_dict[node_class][m]['states'] = 2
                if 'connections' not in self.node_dict[node_class][m]: self.node_dict[node_class][m]['connections'] = 0
                if 'target' not in self.node_dict[node_class][m]: self.node_dict[node_class][m]['target'] = bias_target
                if 'mass' not in self.node_dict[node_class][m]: self.node_dict[node_class][m]['mass'] = 0.0
                if 'polarity' not in self.node_dict[node_class][m]: self.node_dict[node_class][m]['polarity'] = 1
                if 'period' not in self.node_dict[node_class][m]: self.node_dict[node_class][m]['period'] = [1]
                if 'part' not in self.node_dict[node_class][m]: self.node_dict[node_class][m]['part'] = 'any'
                if 'node_ef' not in self.node_dict[node_class][m]: self.node_dict[node_class][m]['node_ef'] = 1.0
                if 'threshold' not in self.node_dict[node_class][m]: self.node_dict[node_class][m]['threshold'] = 0.1
                if 'recur' not in self.node_dict[node_class][m]: self.node_dict[node_class][m]['recur'] = 0

        # set synapse defaults if not already specified
        for class1 in self.node_class_list_dict['ordered']:
            for class2 in self.node_class_list_dict['ordered']:
                if 'connect' not in self.synapse_dict[class1][class2]: self.synapse_dict[class1][class2]['connect'] = False
                if 'type' not in self.synapse_dict[class1][class2]: self.synapse_dict[class1][class2]['type'] = 'fail'
                if 'synapse_ef' not in self.synapse_dict[class1][class2]: self.synapse_dict[class1][class2]['synapse_ef'] = 1
                if 'noise' not in self.synapse_dict[class1][class2]: self.synapse_dict[class1][class2]['noise'] = True
                if 'bound' not in self.synapse_dict[class1][class2]: self.synapse_dict[class1][class2]['bound'] = 1000
                if 'size_mass' not in self.synapse_dict[class1][class2]: self.synapse_dict[class1][class2]['size_mass'] = 0
                if 'change_mass' not in self.synapse_dict[class1][class2]: self.synapse_dict[class1][class2]['change_mass'] = 0

                
# nodes counts by class / type
        self.all_nodes = sum([self.node_dict[key][m]['quantity'] for key in self.node_dict for m in self.node_dict[key]])
        self.network_nodes = sum([self.node_dict[key][m]['quantity'] for key in self.node_class_list_dict['network'] for m in self.node_dict[key]])
        self.logic_nodes = sum([self.node_dict[key][m]['quantity'] for key in self.node_class_list_dict['logic'] for m in self.node_dict[key]])

# build node placement tests
        self.placement_test = {}
        ordered_key_list = []
        for key in self.node_class_list_dict['ordered']:
            if sum([self.node_dict[key][m]['quantity'] for m in self.node_dict[key]]) != 0: ordered_key_list.append(key)
        prior_key_list = []
        for key in ordered_key_list:
            self.placement_test[key] = self.synapse_dict[key][key]['connect'] and all([self.synapse_dict[key][prior_key]['connect'] for prior_key in prior_key_list])
            prior_key_list.append(key)

# Check and refine network parameters
        if self.network not in self.network_type_list:
            print('\n**********   network type error - execution terminated    ****************\n')
            sys.exit() 
        self.edge = int(round(self.all_nodes**(1/self.dimension)))
        if self.edge**self.dimension != self.all_nodes:
            print('\n**********   node quantity error - execution terminated    ****************\n')
            sys.exit()
            
# Build node placement parameters for network build
        if self.dimension == 2:
            self.neighbor_range_list_dict = {}
            if self.network == 'neighbor':
                self.max_connections = max([self.node_dict[key][m]['connections'] for key in self.node_dict for m in self.node_dict[key] if self.node_dict[key][m]['quantity'] > 0])            
                if self.bipartite:
                    r = int(math.sqrt(self.max_connections))-1
                    self.max_neighbor_range_list = list(range(1, r+1, 2))
                else:
                    r = (int(math.sqrt(2 * self.max_connections + 1)) - 1) // 2
                    self.max_neighbor_range_list = list(range(1, r+1))

                for key in self.node_dict:
                    self.neighbor_range_list_dict[key] = {}
                    for m in self.node_dict[key]:
                        if self.node_dict[key][m]['quantity'] > 0:
                            if self.bipartite:
                                r = int(math.sqrt(self.node_dict[key][m]['connections'])) - 1
                                if (r+1)**2 != self.node_dict[key][m]['connections']:
                                    print(self.node_dict[key][m]['connections'], r, (r+1)**2)
                                    print('\n****************   "connections" parameter incorrect for bipartite neighbor node placement - execution terminated     ****************\n')
                                    sys.exit()
                                else:
                                    self.neighbor_range_list_dict[key][m] = list(range(1, r+1, 2)) 
                            else:
                                r = (int(math.sqrt(2 * self.node_dict[key][m]['connections'] + 1)) - 1) // 2
                                if 2*r*(r+1) != self.node_dict[key][m]['connections']:
                                    print('\n****************   "connections" parameter incorrect for non-bipartite neighbor node placement - execution terminated     ****************\n')
                                    sys.exit()
                                else:
                                    self.neighbor_range_list_dict[key][m] = list(range(1, r+1))
            else:
                self.max_neighbor_range_list = [1]
                self.neighbor_range_list_dict = {key : {m : [1] for m in self.node_dict[key]} for key in self.node_dict}
                    
        self.scale = float(self.scale)
        self.time = int(self.time)
        self.epochs = int(self.epochs)
        self.print_records = int(self.print_records)
                    
# Create folder to store simulations if one does not already exist
        if not os.path.exists('simulations'): os.mkdir('simulations')

# Create folder name string for result storage
        if self.network == 'neighbor': network_name = 'Nei'
        if self.network == 'random': network_name = 'Ran'
        if self.network == 'gaussian': network_name = 'Gau'
        if self.network == 'exponential': network_name = 'Exp'
        network_string = str(max([self.node_dict['discrete'][i]['connections'] for i in self.node_dict['discrete'] if self.node_dict['discrete'][i]['quantity']>0]))
        network_string += network_name + '-' + ('2P' if self.bipartite else '1P')
        recur_string = str(max([self.node_dict['discrete'][i]['recur'] for i in self.node_dict['discrete'] if self.node_dict['discrete'][i]['quantity']>0]))
        network_node_string = str(self.network_nodes) + 'Rc' + recur_string + 'Net'
        logic_node_string = str(self.logic_nodes) + 'Bia'        
        threshold_string = str(max([self.node_dict['discrete'][i]['threshold'] for i in self.node_dict['discrete'] if self.node_dict['discrete'][i]['quantity']>0])) + 'Thr'        
        length_string = str(self.time * self.epochs) + 'Stp'
        node_energy_factor = max([self.node_dict['discrete'][m]['node_ef'] for m in self.node_dict['discrete'] if self.node_dict['discrete'][m]['quantity']>0])
        node_energy_string = str(node_energy_factor) + 'Nf'
        synapse_energy_string = str(self.synapse_dict['discrete']['discrete']['synapse_ef']) + 'Sf'        
        time_string = tm.strftime('%Y%h%d-%H%M%S')
        self.folder_name = 'simulations' + '\\' + network_string + '-' + network_node_string + '-' + logic_node_string + '-' + threshold_string + '-'
        self.folder_name += length_string + '-' + node_energy_string + '-' + synapse_energy_string +  '-' + time_string + qualifier_string
        os.mkdir(self.folder_name)
        print('directory ' + self.folder_name + ' created to store simulation output')

        
        '''
        Simulation output parameters
        :param self.save_relaxation_states: saves node states during MCMC network simulation for visualization in video
        :param self.print_records: saves network state information of last n steps and queues printer output, 0 means no printer output
        :param self.show_video: displays state video frames as they are computed
        :param self.save_video: saves state and state-change videos to disk
        :param self.save_images: saves every nth state image as png file, 0 means don't save
        :param self.save_plots: saves summary plots of the network statistics 

        Network type list and component class dictionaries
        :param self.network_type_list = ['neighbor', 'random', 'gaussian', 'exponential'] 
        :param self.node_class_list_dict = {}
        :param self.node_class_list_dict['ordered'] = ['bias', 'discrete']
        :param self.node_class_list_dict['network'] = ['discrete']
        :param self.node_class_list_dict['logic'] = ['bias']
        :param self.node_class_list_dict['compound'] = ['network', 'logic']
        :param self.node_label_dict = {'network':'Network', 'logic':'Logic'}

        Network Execution Parameters
        :param temperature: network thermal bath temperature
        :param time: Number of time steps in one epoch of the simulation
        :param epochs: Number of epochs in the simulation
        :param period: Length of one period if the simulation includes driven (e.g. logic) nodes that change state periodically.  period <= 1 is equivalent to no period
        :param learn_time: Length of time at the beginning of the simulation where the synapse weights are allowed to adapt.  For later times the synapses have fixed weights.
        :param records: Number of time steps to retain in network, node and synapse histories
        :param reconnect: boolean specifying whether the network should evolve weak connections - True / False means that the network should / should not make reconnections

        Network Architecture Parameters
        :param dimension: Dimension of the network grid
        :param network: A string specifying the kind of network connectivity - 'neighbor', 'random', 'gaussian', 'exponential'
        :param metric: A string specifying the metric to use when determining connectivity - 'constant', 'manhattan'
        :param scale: Characteristic length scale for determining connectivity
        :param omega: Synapse weight scale parameter to limit synapse growth

        Node Description Parameters
        :param node_dict: A dictionary storing the node descriptions keyed by the node type - 'binary', 'ternary' 'noise', 'or', 'and', 'bias1', 'bias2' 
        :param quantity: Integer specifying number of nodes of type specified in the dictionary key
        :param connections: Integer specifying number of connections to the nodes of type specified in the dictionary key
        :param weight: A string specifing the weight type for the node - 'fixed', 'realpn1','realpn2', 'real1', 'real2'
        :param target: A real specifying a target source / sink charge or value / scale of the weights to the node
        :param flag: A string used configure the polarity of the weights of the node - 'none', 'alt', 'rand'
        :param rule: A string indicating the rule to distribute errors to the synapses - 'none', 'proportionate', 'uniform'

        Synapse Description Parameters
        :param size_mass: a real specifying the mass associated with weight growth (causes weight decay)
        :param change_mass: a real specifying the mass associated with weight change (sets a learning rate - higher is slower)

        Network Descriptors
        :param node_class_list_dict: a dictionary of node type lists grouping nodes into categories keyed as 'ordered', 'network', 'logic', 'compound'
        :param synapse_type_list_dict: a dictionary of synapse type lists grouping synapses into categories as 'ordered', 'plastic', 'compound'
        :param all_nodes: an integer specifying the total number of nodes in the simulation - total must fit on a square grid of specified dimension
        :param all_synapses: an integer specifying the total number of synapses
        :param edge: an integer specifying the number of nodes on each edge of the grid
        :param scale: a length scale parameter used to compute distance between nodes
        '''








