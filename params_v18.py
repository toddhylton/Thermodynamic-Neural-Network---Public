# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# This software is Copyright © 2019 The Regents of the University of California. All Rights Reserved. Permission to copy, modify, and distribute this software and its documentation for educational, research and
# non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice, this paragraph and the following three paragraphs appear in all copies. Permission
# to make commercial use of this software may be obtained by contacting:
#
# Office of Innovation and Commercialization
# 9500 Gilman Drive, Mail Code 0910
# University of California
# La Jolla, CA 92093-0910
# (858) 534-5815
# invent@ucsd.edu

# This software program and documentation are copyrighted by The Regents of the University of California. The software program and documentation are supplied “as is”, without any accompanying services from The Regents.
# The Regents does not warrant that the operation of the program will be uninterrupted or error-free. The end-user understands that the program was developed for research purposes and is advised not to rely exclusively
# on the program for any reason.

# IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS
# DOCUMENTATION, EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN “AS IS” BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT,
# UPDATES, ENHANCEMENTS, OR MODIFICATIONS
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Thermodynamic Neural Network Description

import sys
import math
import time as tm
import os

class Parameters(object):
    def __init__(self):

# file versions and name qualifier suffix
        self.network_version = '18'
        self.nodes_version = '18'
        self.synapse_version = '18'
        self.params_version = '18'
        self.render_version = '18'
        qualifier_string = ''

# simulation output parameters
        self.save_relaxation_states = False
        self.print_records = 0
        self.show_video = False
        self.save_state_video = True
        self.save_change_video = False
        self.save_images = 50
        self.save_plots = True

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
        self.relax_cycles = 10
        self.era = {}
        self.era[0]  =  {'epochs': 20,      'logic_mode': 'driven',     'weight_update': True,      'predict_mode': False}
        self.era[1]  =  {'epochs': 00,      'logic_mode': 'off',        'weight_update': False,     'predict_mode': False}
        self.era[2]  =  {'epochs': 00,      'logic_mode': 'driven',     'weight_update': False,     'predict_mode': True}
        self.epochs = sum([self.era[i]['epochs'] for i in self.era])

# network architecture parameters
        self.dimension = 2
        self.network = 'neighbor'
        self.scale = 1.0
        self.bipartite = True
        self.node_recur = 0
        self.bias_node_placement_separation = 10  #20   # 8 for 16 in 1600 /  24 for 8 in 10000 for 2D neighbor networks with 4 connections (or randomly connected networks) / 24 for 32 in 40k 2D neighbor networks / 16 for 32 in 10k neighbor networks
        self.bias_node_link_separation = 5   # 5 for 16 in 1600 for 2D neighbor networks with 4 connections (or randomly connected networks)

# node description parameters
        self.node_dict = {class1: {} for class1 in self.node_class_list_dict['ordered']}

        # network node classes
        self.node_dict['discrete'][0]   =   {'quantity': 000,     'states': 2,    'connections': 4,   'mass': 0,    'node_ef': 1}
        self.node_dict['discrete'][1]   =   {'quantity': 000,     'states': 3,    'connections': 4,   'mass': 0,    'node_ef': 1}
        self.node_dict['discrete'][2]   =   {'quantity': 892,     'states': 101,  'connections': 4,   'mass': 0,    'node_ef': 1}

        # logic node classes
        bias_target = 200.0

        self.node_dict['bias'][0] =         {'quantity': 2,     'complement':True,     'connections': 4,    'period': [13],   'part': 'even'}
        self.node_dict['bias'][1] =         {'quantity': 2,     'complement':True,     'connections': 4,    'period': [7],  'part': 'odd'}
        self.node_dict['bias'][2] =         {'quantity': 2,     'complement':True,     'connections': 4,    'period': [21],  'part': 'even'}
        self.node_dict['bias'][3] =         {'quantity': 2,     'complement':True,     'connections': 4,    'period': [11],  'part': 'odd'}

        self.node_dict['bias'][4] =         {'quantity': 0,     'complement':True,     'connections': 4,    'period': [17],  'part': 'even'}
        self.node_dict['bias'][5] =         {'quantity': 0,     'complement':True,     'connections': 4,    'period': [19],  'part': 'odd'}
        self.node_dict['bias'][6] =         {'quantity': 0,     'complement':True,     'connections': 4,    'period': [23],  'part': 'even'}
        self.node_dict['bias'][7] =         {'quantity': 0,     'complement':True,     'connections': 4,    'period': [29],  'part': 'odd'}

        self.node_dict['bias'][8] =         {'quantity': 0,     'complement':True,     'connections': 4,    'period': [22],  'part': 'even'}
        self.node_dict['bias'][9] =         {'quantity': 0,     'complement':True,     'connections': 4,    'period': [25],  'part': 'odd'}
        self.node_dict['bias'][10] =        {'quantity': 0,     'complement':True,     'connections': 4,    'period': [33],  'part': 'even'}
        self.node_dict['bias'][11] =        {'quantity': 0,     'complement':True,     'connections': 4,    'period': [40],  'part': 'odd'}

        self.node_dict['bias'][12] =        {'quantity': 0,     'complement':True,     'connections': 4,    'period': [16],  'part': 'even'}
        self.node_dict['bias'][13] =        {'quantity': 0,     'complement':True,     'connections': 4,    'period': [31],  'part': 'odd'}
        self.node_dict['bias'][14] =        {'quantity': 0,     'complement':True,     'connections': 4,    'period': [37],  'part': 'even'}
        self.node_dict['bias'][15] =        {'quantity': 0,     'complement':True,     'connections': 4,    'period': [43],  'part': 'odd'}


# synapse description parameters
        self.synapse_dict = {class1: {class2: {} for class2 in self.node_class_list_dict['ordered']} for class1 in self.node_class_list_dict['ordered']}

        # set connection specifics for allowable connections
        self.synapse_dict['discrete']['discrete'] =     {'connect': True,    'type': 'real1',   'synapse_ef':40}
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
                if 'polarity' not in self.node_dict[node_class][m]: self.node_dict[node_class][m]['polarity'] = 1
                if 'period' not in self.node_dict[node_class][m]: self.node_dict[node_class][m]['period'] = [1]
                if 'part' not in self.node_dict[node_class][m]: self.node_dict[node_class][m]['part'] = 'any'
                if 'node_ef' not in self.node_dict[node_class][m]: self.node_dict[node_class][m]['node_ef'] = 1.0

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

# build node placement tests - helps node placement algorithm
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

# Build node placement parameters for network build - computes parameters for node placements and connections used in 2D network construction
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
        network_node_string = str(self.network_nodes) + 'Rc' + str(self.node_recur) + 'Net'
        logic_node_string = str(self.logic_nodes) + 'Bia'
        mc_string = str(self.relax_cycles) + 'MC' + ('rec' if self.save_relaxation_states else '')
        length_string = str(self.time * self.epochs) + 'Stp'
        node_energy_factor = max([self.node_dict['discrete'][m]['node_ef'] for m in self.node_dict['discrete'] if self.node_dict['discrete'][m]['quantity']>0])
        node_energy_string = str(node_energy_factor) + 'Nf'
        synapse_energy_string = str(self.synapse_dict['discrete']['discrete']['synapse_ef']) + 'Sf'
        time_string = tm.strftime('%Y%h%d-%H%M%S')
        self.folder_name = 'simulations' + '\\' + network_string + '-' + network_node_string + '-' + logic_node_string + '-' + mc_string + '-' + length_string + '-' + node_energy_string + '-' + synapse_energy_string +  '-' + time_string + qualifier_string
        os.mkdir(self.folder_name)
        print('directory ' + self.folder_name + ' created to store simulation output')



        '''
        Simulation output parameters
        :param self.save_relaxation_states: if True saves node states during MCMC network simulation for visualization in video
        :param self.print_records: saves network state information of last n steps and queues printer output, 0 means no printer output
        :param self.show_video: if True displays state video frames as they are computed
        :param self.save_state_video: if True saves state videos to disk
        :param self.save_state_video: if True saves state change videos to disk
        :param self.save_images: integer specifying period between saving video frames a indidual image file.  e.g. if self.save_images = 10 then every 10th frame is saved
        :param self.save_plots: if True saves plots of network evolution statistics

        Network type list and component class dictionaries
        :param self.network_type_list = ['neighbor', 'random', 'gaussian', 'exponential'] - specifies network connectivity type
        :param self.node_class_list_dict = {} - specifies different types of nodes at lists in a dictionary
        :param self.node_class_list_dict['ordered'] = ['bias', 'discrete'] - a list of node types in the order that they are placed in the network build
        :param self.node_class_list_dict['network'] = ['discrete'] - a list of network node types
        :param self.node_class_list_dict['logic'] = ['bias'] - a list of logic or bias node types
        :param self.node_class_list_dict['compound'] = ['network', 'logic'] - a list of compound node list types
        :param self.node_label_dict = {'network':'Network', 'logic':'Logic'} - labels use in some output functions

        Network Execution Parameters
        :param self.time: number of time steps in an epoch
        :param self.relax_cycles: number of round-robin relaxation cycles per time step
        :param self.era: dictionary of execution parameters for larger scale network execution changes
            :param 'epochs': number of epochs in a era
            :param 'logic_mode': specifies how the logic nodes select state during an epoch
                'driven' - logic node change independent of its inputs
                'off' - logic nodes have zero output
                'noise' - logic nodes create noisy output
                'reflect' - logic nodes choose to state to best match it inputs
            :param 'weight_update': if True weights are updated with errors from nodes, otherwise they are not updated
            :param 'predict_mode': if False logic nodes generate programmed output regardless of their inputs, if True logic nodes generate programmed output only for inputs that correctly predict that output and generate no output otherwise
        :param self.epochs: sums total number of epochs across eras

        Network Architecture Parameters
        :param self.dimension: Dimension of the network grid - typically 2, higher dimensions may work but the code will likely need modifications
        :param self.network: A string specifying the kind of network connectivity - 'neighbor', 'random', 'gaussian', 'exponential'
        :param self.scale: Characteristic length used to determine connection probability in gaussian and exponential networks
        :param self.bipartite:
            if True, nodes are segmented into two partitions and nodes are allowed to connect only to nodes in the opposite partition; also inverts the polarity of the display in one partition so that order can be viewed more easily
            if False, nodes are all in one partition; for nearest neighbor networks, which are inherently bipartitioned, False turns off the display inversion
        :param self.node_recur: integer specifying the number of recurrent connection for each network node type
        :param self.bias_node_placement_separation: integer specifying minimum number network nodes that must separate logic / bias nodes on the network grid - creates separation between nodes in neighbor networks (and separation in node display position for non-neighbor networks)
        :param self.bias_node_link_separation:  integer specifying minimum number of connection hops that must separate logic / bias nodes - creates separation between nodes in non-neighbor networks

        Node Description Parameters
        :param node_dict: A dictionary storing the node descriptions keyed by the node type
            :param 'quantity': Integer specifying number of nodes of type specified in the dictionary key
            :param 'connections': Integer specifying number of connections to the nodes of type specified in the dictionary key'
            :param 'complement': Boolean specifying if bias nodes have alternating bias and partition designations
            :param 'states': Integer specifying number of node state on the interval [-1,1]
            :param 'target': Real specifying bias node output weight values or adaptation targets
            :param 'polarity': +/- 1 sets polarity of bias nodes
            :param 'period': integer sets period for bias node polarity change
            :param 'part':  restricts bias node placement in bipartitioned networks - can be 'even', 'odd', or 'any'
            :param 'node_ef': sets inverse temperature for network node state selection


        Synapse Description Parameters
        :param self.synapse_dict: dictionary specifying synapse types
            :param self.synapse_dict[class1][class2]: dictionary entry specifying type of connection between different nodes of type 'class1' and 'class2'
                :param 'connect': boolean specifying whether connections are allowed between node types
                :param 'type': string specifying synapse type - 'real1', 'real2', 'fixed', 'fail'
                :param 'synapse_ef': real specifying synapse inverse temperature in weight update
                :param 'noise': boolean to turn on (True) / off (False) noise in the weight update
                :param 'bound': real setting maximum absolute value of the synapse weight - usually set to well exceed maximum weight values
                :param 'size_mass': real specifying the mass associated with weight growth (causes weight decay) - seldom used
                :param 'change_mass': real specifying the mass associated with weight change (sets a learning rate - higher is slower) - seldom used
        '''
