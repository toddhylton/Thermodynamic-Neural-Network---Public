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


import numpy as np
import copy as cp
import sys
import math
import time as tm
import os
import shutil
import params_v18 as pd
import nodes_v18 as nd
import synapse_v18 as sd
import render_v18 as rd


class Network(object):
    '''
    Network class for nodes arranged on a regular grid connected periodically so there are no boundaries.
    '''

    def __init__(self):

#       import the network definition
        self.parm = pd.Parameters()

#       open and initialize file to store network state data
        self.data_dir = self.parm.folder_name + '\\data'
        os.mkdir(self.data_dir)
        self.state_filename = self.data_dir + '\\network_data.txt'
        self.state_file = open(self.state_filename, 'w')
        print('file "' + self.state_filename + '" created to store simulation state data')

        self.state_file.write('Thermodynamic Neural Network Data File\n')
        self.state_file.write('Date %s\n' %tm.strftime('%m/%d/%Y'))
        self.state_file.write('Time %s\n' %tm.strftime('%H:%M:%S'))

        self.state_file.write('\nVersion Information\n')
        self.state_file.write('network module = ' + self.parm.network_version + '\n')
        self.state_file.write('simulation specification module = ' + self.parm.params_version + ' ' + str(pd) + '\n')
        self.state_file.write('node module = ' + self.parm.nodes_version + ' ' + str(nd) + '\n')
        self.state_file.write('synapse module = ' + self.parm.synapse_version + ' ' + str(nd) + '\n')
        self.state_file.write('render module = ' + self.parm.render_version + ' ' + str(rd) + '\n')

        self.state_file.write('\nNetwork Execution Parameters\n')
        self.state_file.write('time steps per epoch = %5i\n' %self.parm.time)
        self.state_file.write('relaxation cycles = %5i\n' %self.parm.relax_cycles)
        self.state_file.write('save relaxation states = %s\n' %self.parm.save_relaxation_states)
        self.state_file.write('\nExecution Eras\n')
        for i in self.parm.era:
            if self.parm.era[i]['epochs'] > 0: self.state_file.write('era = %2i params = %s\n' %(i, self.parm.era[i]))

        self.state_file.write('\nNetwork Architecture Parameters\n')
        self.state_file.write('dimension = %2i\n' %self.parm.dimension)
        self.state_file.write('total nodes = %4i\n' %self.parm.all_nodes)
        self.state_file.write('total network nodes = %4i\n' %self.parm.network_nodes)
        self.state_file.write('total logic nodes = %4i\n' %self.parm.logic_nodes)
        self.state_file.write('network type = %s\n' %self.parm.network)
        self.state_file.write('scale = %2.2f\n' %self.parm.scale)
        self.state_file.write('bipartite = %s\n' %self.parm.bipartite)
        self.state_file.write('node recurrence = %1i\n' %self.parm.node_recur)
        self.state_file.write('minimum bias node placement separation length = %2i\n' %self.parm.bias_node_placement_separation)
        self.state_file.write('minimum bias node link separation length = %2i\n' %self.parm.bias_node_link_separation)

        self.state_file.write('\nNode Parameters\n')
        for key in self.parm.node_dict:
            for m in self.parm.node_dict[key]:
                if self.parm.node_dict[key][m]['quantity'] != 0:
                    self.state_file.write('type = %s  params =%s\n' %(key, self.parm.node_dict[key][m]))

        self.state_file.write('\nSynapse Parameters\n')
        for class1 in self.parm.synapse_dict:
            for class2 in self.parm.synapse_dict[class1]:
                self.state_file.write('connection type = %s to %s  params =%s\n' %(class1, class2, self.parm.synapse_dict[class1][class2]))

#       open and initialize file to store network plotting data
        self.plot_filename = self.data_dir + '\\plot_data.txt'
        self.plot_file = open(self.plot_filename, 'w')
        print('file "' + self.plot_filename + '" created to store simulation plotting data')
        self.plot_file.write('Thermodynamic Neural Network Plotting File\n')
        self.plot_file.write('Date %s12\n' %tm.strftime('%m/%d/%Y'))
        self.plot_file.write('Time %s12\n' %tm.strftime('%H:%M:%S'))

        print('\n*********************************************  Building Network  ****************************************************\n')

#       Populate the positional information of nodes on the grid dimensions in the array node_position_array
        self.all_node_list = list(range(self.parm.all_nodes))
        all_node_array = np.array(self.all_node_list)
        dimension_list = list(range(self.parm.dimension))
        node_position_array = np.zeros((self.parm.all_nodes, self.parm.dimension),dtype=int)
        for i in self.all_node_list:
            position = i
            for d in dimension_list:
                factor = self.parm.edge**(self.parm.dimension-d-1)
                node_position_array[i,d] = position//factor
                position = position % factor
        even_node_list = [i for i in self.all_node_list if np.sum(node_position_array[i])%2 == 0]
        odd_node_list = [i for i in self.all_node_list if np.sum(node_position_array[i])%2 == 1]
        if self.parm.bipartite:
            self.display_polarity = np.array([1 - 2*(np.sum(node_position_array[i])%2) for i in self.all_node_list], dtype=int)
            polarity = cp.copy(self.display_polarity)
        else:
            self.display_polarity = np.ones(self.parm.all_nodes, dtype=int)

        print('\n**********  %s network node position completed  *****************\n' %self.parm.network)

#       Compute the distance between any two nodes on a grid and store the distance in the array metric_array
        metric_array = np.zeros((self.parm.all_nodes,self.parm.all_nodes), dtype=int)
        neighbor_dict = {}
        for i in self.all_node_list:
            distance_array = np.abs(node_position_array[i] - node_position_array)
            distance_boolean = 2 * distance_array < self.parm.edge
            distance_array = distance_array * distance_boolean + (self.parm.edge - distance_array) * ~distance_boolean
            metric_array[i] = np.sum(distance_array,1)
            neighbor_mask = np.zeros(self.parm.all_nodes, dtype=bool)
            for d in self.parm.max_neighbor_range_list:
                neighbor_mask = neighbor_mask | (metric_array[i] == d)
            neighbor_dict[i] = list(all_node_array[neighbor_mask])

#       Initiate node data structures
        available_even_node_list = cp.copy(even_node_list)
        available_odd_node_list = cp.copy(odd_node_list)
        self.node_class = {i:'none' for i in self.all_node_list}
        node_target = [0 for i in self.all_node_list]
        node_connect = np.array([0 for i in self.all_node_list], dtype=int)
        self.node_list_dict = {}
        self.node = [0 for i in self.all_node_list]
        bias_node_link_separation = [self.parm.edge for i in self.all_node_list]
        nodes_placed = 0

        print('\n**********  %s network node distance completed  *****************\n' %self.parm.network)

#       Place nodes as constrained by the node and synapse dictionaries and network specifications
        for key in self.parm.node_class_list_dict['ordered']:
            self.node_list_dict[key] = []
            for m in self.parm.node_dict[key]:
                for n in range(self.parm.node_dict[key][m]['quantity']):
                    complement = self.parm.node_dict[key][m]['complement'] and n%2 == 1

#                   Setup a node placement test
                    test = False
                    count = 0
                    viable_even_node_list = cp.copy(available_even_node_list)
                    viable_odd_node_list = cp.copy(available_odd_node_list)
                    while not test:

#                       Randomly select a placement position for the node.
                        if self.parm.bipartite or self.parm.network == 'neighbor':
                            if complement:
                                if self.parm.node_dict[key][m]['part'] == 'any': i = np.random.choice(viable_even_node_list + viable_odd_node_list)
                                if self.parm.node_dict[key][m]['part'] == 'even': i = np.random.choice(viable_even_node_list)
                                if self.parm.node_dict[key][m]['part'] == 'odd': i = np.random.choice(viable_odd_node_list)
                            else:
                                if self.parm.node_dict[key][m]['part'] == 'any': i = np.random.choice(viable_even_node_list + viable_odd_node_list)
                                if self.parm.node_dict[key][m]['part'] == 'even': i = np.random.choice(viable_odd_node_list)
                                if self.parm.node_dict[key][m]['part'] == 'odd': i = np.random.choice(viable_even_node_list)
                        else:
                            i = np.random.choice(viable_even_node_list + viable_odd_node_list)


#                       Test node position for separation from prohibited neighbor nodes that are already placed
                        test = True
                        i_neighbor_list = [j for j in self.all_node_list if metric_array[i,j] in self.parm.neighbor_range_list_dict[key][m]]
                        if not self.parm.placement_test[key]:
                            i_list = cp.copy(i_neighbor_list)
                            test_list = [k for k in i_list if self.node_class[k] != 'none']
                            test = all([self.parm.synapse_dict[key][self.node_class[k]]['connect'] for k in test_list])
                            p = 1
                            while test and p <= self.parm.bias_node_placement_separation:
                                if len(i_list) < self.parm.all_nodes:
                                    i_list = list(set(i_list + [j for k in i_list for j in neighbor_dict[k]]))
                                else:
                                    i_list = self.all_node_list
                                test_list = [k for k in i_list if self.node_class[k] != 'none']
                                test = all([self.parm.synapse_dict[key][self.node_class[k]]['connect'] for k in test_list])
                                p += 1
                            if not test:
                                print('placement conflict in neighbor nodes detected', count)
                                for q in test_list:
                                    print('separation', p, 'distance', metric_array[i,q])
                                count += 1
                                if i in viable_even_node_list: viable_even_node_list.remove(i)
                                if i in viable_odd_node_list: viable_odd_node_list.remove(i)
                                print(len(available_even_node_list), len(viable_even_node_list), len(available_odd_node_list), len(viable_odd_node_list))
                                if len(viable_even_node_list)==0 or len(viable_odd_node_list)==0:
                                    print('neighbor node placement failure - execution terminated')
                                    self.kill_simulation()

#                   Update node lists after node placement
                    nodes_placed += 1
                    self.node_list_dict[key].append(i)
                    if i in available_even_node_list: available_even_node_list.remove(i)
                    if i in available_odd_node_list: available_odd_node_list.remove(i)
                    for j in neighbor_dict[i]:
                        if j not in i_neighbor_list: neighbor_dict[j].remove(i)
                    neighbor_dict[i] = cp.copy(i_neighbor_list)

#                   Assign node description parameters to the selected node position
                    self.node_class[i] = key
                    node_states = self.parm.node_dict[key][m]['states']
                    node_target[i] = self.parm.node_dict[key][m]['target']
                    node_connect[i] = self.parm.node_dict[key][m]['connections']
                    node_polarity = self.parm.node_dict[key][m]['polarity'] * (1 - 2 * int(complement))
                    node_period = self.parm.node_dict[key][m]['period']
                    node_energy_factor = self.parm.node_dict[key][m]['node_ef']

#                   Build the node object
                    self.node[i] = nd.MakeNode.Factory(i, key, node_states, node_polarity, node_period, self.parm.print_records, node_energy_factor)
                    if nodes_placed % 1000 == 0: print('%i nodes placed' %nodes_placed)

#                   Set up logic node separation tracking for synapse placement in non-neighbor networks
                    if key in self.parm.node_class_list_dict['logic']: bias_node_link_separation[i] = 0

        print('\n**********  %s network %i node placement completed  ************\n' %(self.parm.network, nodes_placed))

#       Define compound node type lists in the dictionary
        for key1 in self.parm.node_class_list_dict['compound']:
            self.node_list_dict[key1] = []
            for key2 in self.parm.node_class_list_dict[key1]:
                self.node_list_dict[key1] += self.node_list_dict[key2]

#       Correct display polarity for logic nodes
        for i in self.node_list_dict['logic']: self.display_polarity[i] = 1

#       Write node type and position information to the network output file
        line = '\nNode Index, Type, dim ' + ', dim '.join([str(d+1) for d in dimension_list]) + '\n'
        self.state_file.write(line)
        for i in self.all_node_list:
            line = str(i) + ', ' + self.node[i].node_type + ', ' + ', '.join([str(node_position_array[i,d]) for d in dimension_list]) + '\n'
            self.state_file.write(line)

#       Create network data structures
        self.node_to_synapse_list = [[] for i in self.all_node_list]                                    # initialize a dictionary of node-to-synapse connection lists
        self.node_to_node_list = [[] for i in self.all_node_list]                                       # initialize a dictionary of node-to-node connection lists
        self.synapse_key_map = {}                                                                       # initiate a dictionary to index nodes from from synapse index
        self.network_synapse_list = []
        synapse_depth = {}

#       Create network graph for neighbor networks
        if self.parm.network == 'neighbor':
            k = 0
            for i in self.all_node_list:
                for j in neighbor_dict[i]:
                    if i in neighbor_dict[j]:
                        if (self.parm.node_class_list_dict['ordered'].index(self.node_class[j]) < self.parm.node_class_list_dict['ordered'].index(self.node_class[i])):
                            self.synapse_key_map[k] = (i,j)
                        else:
                            self.synapse_key_map[k] = (j,i)
                        self.network_synapse_list.append(k)
                        synapse_depth[k] = 0
                        self.node_to_synapse_list[i].append(k)
                        self.node_to_synapse_list[j].append(k)
                        self.node_to_node_list[i].append(j)
                        self.node_to_node_list[j].append(i)
                        neighbor_dict[j].remove(i)
                        k += 1
            self.all_synapses = k
            print('%i synapses placed' %k)

#       Create network graph for non-neighbor networks
        if self.parm.network != 'neighbor':

#           Compute probability array and probaility mask of connection between any two nodes
            if self.parm.bipartite: connection_mask = np.subtract(polarity, polarity.reshape(self.parm.all_nodes,1)).astype(bool)
            else: connection_mask = ~np.eye(self.parm.all_nodes, dtype=bool)
            if self.parm.network == 'random': probability_array = np.ones((self.parm.all_nodes, self.parm.all_nodes), dtype=bool)
            if self.parm.network == 'gaussian': probability_array = np.exp(-metric_array**2/self.parm.scale**2/2.0)
            if self.parm.network == 'exponential': probability_array = np.exp(-metric_array/self.parm.scale)
            saturation_mask = np.ones(self.parm.all_nodes, dtype=bool)
            mask_sum = np.sum(connection_mask,1)

#           Find edges in the graph
            k = 0                                           # synapse / connection counter
            f = 0                                           # connection failure counter

#           Randomly select a node to connect via binary search
            i_search = True
            while i_search:
                cum_connect = np.cumsum(node_connect)
                seed = np.random.random() * cum_connect[-1]
                i = self.parm.all_nodes//2
                imin = 0
                imax = self.parm.all_nodes
                while (imax != imin+1):
                    if seed > cum_connect[i-1]: imin = i
                    else: imax = i
                    i = (imax + imin)//2

                mask_sum_test = cp.copy(mask_sum)
                if node_connect[i]==1: mask_sum_test -= connection_mask[i]
                else: mask_sum_test[i] -= 1

#               Randomly select another node to connect via binary search
                j_search = True
                while j_search:
                    cum_probability = np.cumsum(saturation_mask * connection_mask[i] * probability_array[i])
                    seed = np.random.random() * cum_probability[-1]
                    j = self.parm.all_nodes//2
                    jmin = 0
                    jmax = self.parm.all_nodes
                    while (jmax != jmin+1):
                        if seed > cum_probability[j-1]: jmin = j
                        else: jmax = j
                        j = (jmax + jmin)//2

                    if node_connect[j]==1: mask_sum_test -= connection_mask[j]
                    else: mask_sum_test[j] -= 1

#                   test for conflicts with existing connectivity
                    if np.all(mask_sum_test >= node_connect):

#                       test for conflicts with bias node separation
                        if self.parm.bias_node_link_separation <= (bias_node_link_separation[i] + bias_node_link_separation[j] + 1):
                            j_search = False

#                           Update probability masks
                            connection_mask[i,j] = connection_mask[j,i] = False

                            node_connect[i] -= 1
                            if node_connect[i]==0:
                                mask_sum -= connection_mask[i]
                                saturation_mask[i] = False
                            else:
                                mask_sum[i] -= 1

                            node_connect[j] -= 1
                            if node_connect[j]==0:
                                mask_sum -= connection_mask[j]
                                saturation_mask[j] = False
                            else:
                                mask_sum[j] -= 1

#                           Store connection
                            if (self.parm.node_class_list_dict['ordered'].index(self.node_class[j]) < self.parm.node_class_list_dict['ordered'].index(self.node_class[i])):
                                self.synapse_key_map[k] = (i,j)
                            else:
                                self.synapse_key_map[k] = (j,i)
                            self.network_synapse_list.append(k)
                            synapse_depth[k] = 0
                            self.node_to_synapse_list[i].append(k)
                            self.node_to_synapse_list[j].append(k)
                            self.node_to_node_list[i].append(j)
                            self.node_to_node_list[j].append(i)
                            k += 1
                            if k%1000==0: print('%i synapses placed' %k)

#                           Update node separation from logic nodes for future connection tests
                            i_list = [i]
                            j_list = [j]
                            separation = 0
                            stop = False
                            while not stop:
                                stop = True
                                separation += 1
                                for m in cp.copy(i_list):
                                    if bias_node_link_separation[m] > bias_node_link_separation[j] + separation:
                                        bias_node_link_separation[m] = bias_node_link_separation[j] + separation
                                        i_list += [l for l in self.node_to_node_list[m] if l not in cp.copy(i_list)]
                                        i_list.remove(m)
                                        stop = False
                                    else:
                                        i_list.remove(m)
                                for m in cp.copy(j_list):
                                    if bias_node_link_separation[m] > bias_node_link_separation[i] + separation:
                                        bias_node_link_separation[m] = bias_node_link_separation[i] + separation
                                        j_list += [l for l in self.node_to_node_list[m] if l not in cp.copy(j_list)]
                                        j_list.remove(m)
                                        stop = False
                                    else:
                                        j_list.remove(m)

                        else:
#                           kill connections and continue search
                            connection_mask[i,j] = connection_mask[j,i] = False
                            mask_sum[i] -=1
                            mask_sum[j] -=1
                            if mask_sum[i] == 0:
                                j_search = False
                                f += node_connect[i]
                                node_connect[i] = 0
                                print('bias node separation failure for node %i' %i)
                            else:
                                print('bias node separation conflict corrected for nodes (%i,%i)' %(i,j))
                            if mask_sum[j] == 0:
                                f += node_connect[j]
                                node_connect[j] = 0
                                print('bias node separation failure for node %i' %j)
                    else:
#                       kill connections and continue or quit search
                        j_search = False
                        connection_mask[i,j] = connection_mask[j,i] = False
                        mask_sum[i] -=1
                        mask_sum[j] -=1
                        if mask_sum[i] == 0:
                            print('%i connection failure(s) for node %i' %(node_connect[i],i))
                            f += node_connect[i]
                            node_connect[i] = 0
                        if mask_sum[j] == 0:
                            print('%i connection failure(s) for node %i' %(node_connect[j],j))
                            f += node_connect[j]
                            node_connect[j] = 0
                        if mask_sum[i] != 0 and mask_sum[j] != 0:
                            print('connectivity conflict corrected for nodes (%i,%i)' %(i,j))
                        if np.sum(mask_sum) == 0 or np.sum(node_connect) == 0:
                            i_search = False
                            print('%i connections placed in %s network with %i failures' %(k, self.parm.network, f))

            self.all_synapses = k

#       Test for bias node separation errors
        min_separation = self.parm.edge
        for m in self.node_list_dict['logic']:
            separation = 0
            test_list = self.node_to_node_list[m]
            prior_list = [m]
            stop = False
            while not stop:
                stop = any([p in self.node_list_dict['logic'] for p in test_list]) or separation == self.parm.edge
                separation += 1
                prior_list += cp.copy(test_list)
                test_list = list(set([q for p in prior_list for q in self.node_to_node_list[p] if q not in prior_list]))
            min_separation = min(min_separation, separation)
        if min_separation < self.parm.bias_node_link_separation:
            print('logic node separation error in %s network' %self.parm.network)
            for m in self.node_list_dict['logic']:
                print('bias node %i separation = %i' %(m, separation))
            self.kill_simulation()

#       Create recurrent connections
        self.recurrent_synapse_list = []
        k = self.all_synapses
        for m in range(self.parm.node_recur):
            for i in self.node_list_dict['network']:
                self.synapse_key_map[k] = (i,i)
                self.node_to_synapse_list[i].append(k)
                self.recurrent_synapse_list.append(k)
                synapse_depth[k] = 0   # changed
                k += 1
        self.all_synapses = k
        self.recurrent_synapses = len(self.recurrent_synapse_list)

        print('\n*************  %s network %i connections completed   *************\n' %(self.parm.network, self.all_synapses))

#       Build the network
        self.all_synapse_list = list(range(self.all_synapses))
        self.synapse = {}
        self.plastic_synapse_list = []
        for k in self.all_synapse_list:
#           Setup the synapses.
            (i,j) = self.synapse_key_map[k]
            weight_target = node_target[i] + node_target[j]
            weight_bound = self.parm.synapse_dict[self.node_class[i]][self.node_class[j]]['bound']
            weight_type = self.parm.synapse_dict[self.node_class[i]][self.node_class[j]]['type']
            weight_noise = self.parm.synapse_dict[self.node_class[i]][self.node_class[j]]['noise']
            size_mass = self.parm.synapse_dict[self.node_class[i]][self.node_class[j]]['size_mass']
            change_mass = self.parm.synapse_dict[self.node_class[i]][self.node_class[j]]['change_mass']
            energy_factor = self.parm.synapse_dict[self.node_class[i]][self.node_class[j]]['synapse_ef']
            if weight_type == 'fail':
                print('\n**********   network connection error (synapse weight type == fail) - execution terminated    ****************\n')
                self.kill_simulation()
            self.synapse[k] = sd.MakeSynapse.Factory(k, weight_type, energy_factor, synapse_depth[k], weight_bound, weight_target, weight_noise, size_mass, change_mass, self.parm.print_records)
            if weight_type != 'fixed' and self.node[i].node_type != 'bias' and self.node[j].node_type != 'bias': self.plastic_synapse_list.append(k)

#           Connect nodes and synapses
            self.synapse[k].add_nodes(i, self.node[i].receive_context, j, self.node[j].receive_context)
            self.node[i].add_synapse(k, j, weight_type, weight_target, self.synapse[k].update_state, self.synapse[k].push_state)
            if i != j: self.node[j].add_synapse(k, i, weight_type, weight_target, self.synapse[k].update_state, self.synapse[k].push_state)
        self.plastic_synapses = len(self.plastic_synapse_list)

        print('\n**************  %s network build completed   *******************\n' %self.parm.network)


    def run_network(self):

#       print headers
        print('Epoch\t\tAvg Node Energy\t\tAvg Synapse^2\t\t% Changed\t\t% Solved\t\t\tTotal Entropy\t\tAvg Dissipation\t\tAvg Transport\t\tQuality\t\tOrder Param')
        self.state_file.write('\nNetwork Simulation Results\n')
        self.plot_file.write('\nTime, Avg Node Energy, Avg Synapse^2, % Changed, % Solved, Total Entropy, Avg Dissipation, Avg Transport, Quality, Order Param')

#       initialize network history lists
        self.network_long_history = []
        self.network_short_history = []

#       update network in a series of epochs
        time = 0
        epoch = 0
        for era in self.parm.era:
            logic_mode = self.parm.era[era]['logic_mode']           # logic modes can be 'driven' (state set externally), 'noise' (state = random +/- 1), 'off' (state = 0), 'reflect' (state selected from Boltzmann energy distribution)
            weight_update = self.parm.era[era]['weight_update']     # boolean flag for weight updates
            predict_mode = self.parm.era[era]['predict_mode']       # boolean flag for bias nodes to give 0 as output to connected nodes if inputs from those nodes are of incorrect sign during the MCMC sampling stage
            for h in range(self.parm.era[era]['epochs']):
                epoch +=1

#               set initial variables that change over a single epoch
                total_node_energy = 0.0
                total_node_entropy = 0.0
                total_node_dissipation = 0.0
                total_node_transport = 0.0
                total_node_quality_denom = 0.0
                total_node_quality_numer = 0.0
                total_node_solutions = 0
                total_flipped_nodes = 0
                total_synapse_energy = 0.0
                total_synapse_order = 0.0

#               update the network for the current epoch
                for t in range(self.parm.time):
                    time += 1

#                   relax the network node states via a markov chain monte carlo gibbs sampling technique
                    for i in self.node_list_dict['logic']: self.node[i].set_state(time, era, logic_mode)
                    for i in range(self.parm.relax_cycles):
                        for k in self.node_list_dict['logic']: self.node[k].relax_state(predict_mode)
                        for k in self.node_list_dict['network']: self.node[k].relax_state()

#                       save node state information for each step in the Gibbs sampling update if specified in the parameter object defining the network
                        if self.parm.save_relaxation_states:
                            for l in self.node_list_dict['logic']: self.node[l].evaluate_state()
                            self.state_file.write('\ntime, %7i, relaxation cycle, %3i\n' %(time, i))
                            self.state_file.write('node id, energy, state, entropy, solution, dissipation\n')
                            for m in self.all_node_list:
                                line = str(m) + ', ' + str(self.node[m].energy) + ', ' + str(self.display_polarity[m] * self.node[m].state) + ', ' + str(self.node[m].state_change) + ', '
                                line += str(0.0) + ', ' + str(self.node[m].solve) + ', ' + str(0.0) + ', ' + str(0.0) + '\n'
                                self.state_file.write(line)

#                   update the node states / update synapse states
                    for i in self.node_list_dict['network']: self.node[i].update_state(weight_update)
                    for i in self.node_list_dict['logic']: self.node[i].update_state(predict_mode, weight_update)
                    for i in self.node_list_dict['logic']: self.node[i].evaluate_state()
                    self.state_file.write('\ntime, %7i, state update\n' %(time))
                    self.state_file.write('node id, energy, state, entropy, solution, dissipation\n')
                    for i in self.all_node_list:
                        line = str(i) + ', ' + str(self.node[i].energy) + ', ' + str(self.display_polarity[i] * self.node[i].state) + ', ' + str(self.node[i].state_change) + ', '
                        line += str(self.node[i].entropy) + ', ' + str(self.node[i].solve) + ', ' + str(self.node[i].dissipation) + ', ' + str(self.node[i].transport) + '\n'
                        self.state_file.write(line)

#                   update network status variables
                    sum_node_energy = sum([self.node[i].energy for i in self.node_list_dict['network']])
                    sum_node_entropy = sum([self.node[i].entropy for i in self.node_list_dict['network']])
                    sum_node_dissipation = sum([self.node[i].dissipation for i in self.node_list_dict['network']])
                    sum_node_transport = sum([self.node[i].transport for i in self.node_list_dict['network']])
                    sum_node_quality_denom = sum([self.node[i].quality_denom for i in self.node_list_dict['network']])
                    sum_node_quality_numer = sum([self.node[i].quality_numer for i in self.node_list_dict['network']])
                    sum_flipped_nodes = sum([abs(self.node[i].state_change) for i in self.node_list_dict['network']])
                    sum_node_solutions = sum([1 for i in self.node_list_dict['logic'] if self.node[i].solve])
                    sum_synapse_energy = sum([self.synapse[k].weight**2 for k in self.plastic_synapse_list])
                    sum_synapse_order = sum([self.synapse[k].order for k in self.plastic_synapse_list if k not in self.recurrent_synapse_list])

                    total_node_energy += sum_node_energy
                    total_node_entropy += sum_node_entropy
                    total_node_dissipation += sum_node_dissipation
                    total_node_transport += sum_node_transport
                    total_node_quality_denom += sum_node_quality_denom
                    total_node_quality_numer += sum_node_quality_numer
                    total_node_solutions += sum_node_solutions
                    total_flipped_nodes += sum_flipped_nodes
                    total_synapse_energy += sum_synapse_energy
                    total_synapse_order += sum_synapse_order

#                   Store results of the time step
                    avg_node_energy = sum_node_energy / self.parm.network_nodes
                    avg_node_dissipation = sum_node_dissipation / self.parm.network_nodes
                    avg_node_transport = sum_node_transport / self.parm.network_nodes
                    if sum_node_quality_denom == 0.0: avg_node_quality = 0.0
                    else: avg_node_quality = sum_node_quality_numer / sum_node_quality_denom
                    percent_node_solutions = 100.0 * sum_node_solutions / max(1, self.parm.logic_nodes)
                    percent_flipped_nodes = 100.0 * sum_flipped_nodes / self.parm.network_nodes
                    avg_synapse_energy = sum_synapse_energy / self.plastic_synapses
                    avg_synapse_order = sum_synapse_order / (self.plastic_synapses - self.recurrent_synapses)
                    self.network_long_history.append((avg_node_energy, avg_synapse_energy, percent_flipped_nodes, percent_node_solutions, sum_node_entropy, avg_node_dissipation, avg_node_transport, avg_node_quality, avg_synapse_order))
                    self.plot_file.write('\n' + str(time) + ', ' + str(avg_node_energy) + ', ' + str(avg_synapse_energy) + ', ' + str(percent_flipped_nodes) + ', ' + str(percent_node_solutions) + ', ' + str(sum_node_entropy) + ', ' + str(avg_node_dissipation) + ', ' + str(avg_node_transport)+ ', ' + str(avg_node_quality) + ', ' + str(avg_synapse_order))

#               Store and print the results of the epoch
                avg_total_node_energy = total_node_energy / self.parm.network_nodes / self.parm.time
                avg_total_node_entropy = total_node_entropy / self.parm.time
                avg_total_node_dissipation = total_node_dissipation / self.parm.network_nodes / self.parm.time
                avg_total_node_transport = total_node_transport / self.parm.network_nodes / self.parm.time
                if total_node_quality_denom == 0.0: avg_total_node_quality = 0.0
                else: avg_total_node_quality = total_node_quality_numer / total_node_quality_denom
                percent_total_node_solutions = 100.0 * total_node_solutions / max(1, self.parm.logic_nodes) / self.parm.time
                percent_total_flipped_nodes = 100.0 * total_flipped_nodes / self.parm.network_nodes / self.parm.time
                avg_total_synapse_energy = total_synapse_energy / self.plastic_synapses / self.parm.time
                avg_total_synapse_order = total_synapse_order / (self.plastic_synapses - self.recurrent_synapses) / self.parm.time
                self.network_short_history.append((avg_total_node_energy, avg_total_synapse_energy, percent_total_flipped_nodes, percent_total_node_solutions, avg_total_node_entropy, avg_total_node_dissipation, avg_total_node_transport, avg_total_node_quality, avg_total_synapse_order))
                print('%5i\t\t%6.2f\t\t\t%6.2f\t\t\t%4.2f\t\t\t%4.2f\t\t\t\t%4.2f\t\t\t%6.2f\t\t\t%6.2f\t\t\t%6.2f\t\t%6.2f' %(epoch, avg_total_node_energy, avg_total_synapse_energy, percent_total_flipped_nodes, percent_total_node_solutions, avg_total_node_entropy, avg_total_node_dissipation, avg_total_node_transport, avg_total_node_quality, avg_total_synapse_order))

#       print footers and close files
        self.state_file.write('\nEND')
        self.state_file.close()
        self.plot_file.write('\nEND')
        self.plot_file.close()
        print('End Network Simulation on %s at %s' %(tm.strftime('%m/%d/%Y'), tm.strftime('%H:%M:%S')))


    def print_network(self):

        np.set_printoptions(precision=2)

        if input('\nPrint Node Data (y / N)?') == 'y':
            print('\n*****Node Data******')
            for t in range(len(self.node[0].history)):
                print('\ntime=%5d\tNode Type\tConnections\tVoltage\t\tEnergy\t\tEntropy' %t)
                for i in self.all_node_list:
                    print('node=%5d\t%11s\t%5d\t\t%4.2f\t\t%4.2f\t\t%4.4f' %self.node[i].history[t])

        if input('\nPrint Synapse Data (y / N)?') == 'y':
            print('\n*****Synapse Data*****')
            for t in range(len(self.synapse[0].history)):
                print('\ntime=%5d\t\tType\t\tNode 0\t\tNode 1\t\tWeight\t\tPrefactor\tError' %t)
                for k in self.all_synapse_list:
                    print('Synapse%5d\t%9s\t\t%4.2f\t\t%4.2f\t\t%4.2f\t\t%6.3f\t\t%6.3f' %self.synapse[k].history[t])

        for key in self.parm.node_class_list_dict['compound']:
            if self.node_list_dict[key] != []:
                if input('\nPrint ' + self.parm.node_label_dict[key] + ' Node Histories (y / N)?') == 'y':
                    print('\n*****' + self.parm.node_label_dict[key] + ' Node Histories*****')
                    for i in self.node_list_dict[key]:
                        print('\nnode%5d\tNode Type\tConnections\tVoltage\t\tEnergy\t\tEntropy\t\tSolved' %i)
                        for t in range(len(self.node[i].history)):
                            (nh1, nh2, nh3, nh4, nh5, nh6) = self.node[i].history[t]
                            if key == 'logic': nh7 = self.node[i].solution_history[t]
                            else: nh7 = 'na'
                            print('time=%5d\t%11s\t%5d\t\t%4.2f\t\t%4.2f\t\t%4.4f\t\t%5s' %(t, nh2, nh3, nh4, nh5, nh6, nh7))

                if input('\nPrint ' + self.parm.node_label_dict[key] + ' Node Synapse Histories (y / N)?') == 'y':
                    print('\n*****' + self.parm.node_label_dict[key] + ' Node Synapse Histories*****')
                    for j in self.node_list_dict[key]:
                        print('\nNode = %5d\tType = %6s' %(j, self.node_class[j]))
                        for k in self.node_to_synapse_list[j]:
                            print('\nsynapse%5d\t\tType\t\tNode%5d\tNode%5d\tWeight\t\tPrefactor\t\tError\t\tSolved' %(k, self.synapse_key_map[k][1], self.synapse_key_map[k][0]))
                            for t in range(len(self.synapse[k].history)):
                                (sh1, sh2, sh3, sh4, sh5, sh6, sh7) = self.synapse[k].history[t]
                                if key == 'logic': sh8 = self.node[j].solution_history[t]
                                else: sh8 = 'na'
                                print('time=%5d\t%9s\t\t%4.2f\t\t%4.2f\t\t%4.2f\t\t%6.3f\t\t\t%6.3f\t\t%5s' %(t, sh2, sh3, sh4, sh5, sh6, sh7, sh8))

        if input('\nPrint Short History (y / N)?') == 'y':
            print('\n*****Short History******')
            print('\n\t\tAvg Node Energy\t\tAvg Synapse^2\t\t% Changed\t\t% Solved\t\t\tTotal Entropy\t\tAvg Dissipation\t\tAvg Transport\t\tAvg Quality\t\tOrder Param')
            for t in range(len(self.network_short_history)):
                (nh1, nh2, nh3, nh4, nh5, nh6, nh7, nh8, nh9) = self.network_short_history[t]
                print('epoch=%5d\t\t%4.3f\t\t\t%6.2f\t\t\t%4.3f\t\t\t%4.3f\t\t\t%4.4f\t\t\t%4.3f\t\t\t%4.2f\t\t\t%4.2f\t\t\t%4.2f' %(t, nh1, nh2, nh3, nh4, nh5, nh6, nh7, nh8, nh9))

        if input('\nPrint Long History (y / N)?') == 'y':
            print('\n*****Long History*****')
            print('\n\t\tAvg Node Energy\t\tAvg Synapse^2\t\t% Changed\t\t% Solved\t\t\tTotal Entropy\t\tAvg Dissipation\t\tAvg Transport\t\tAvg Quality\t\tOrder Param')
            for t in range(len(self.network_long_history)):
                (nh1, nh2, nh3, nh4, nh5, nh6, nh7, nh8, nh9) = self.network_long_history[t]
                print('time=%5d\t\t%4.3f\t\t\t%6.2f\t\t\t%4.3f\t\t\t%4.3f\t\t\t%4.4f\t\t\t%4.3f\t\t\t%4.2f\t\t\t%4.2f\t\t\t%4.2f' %(t, nh1, nh2, nh3, nh4, nh5, nh6, nh7, nh8, nh9))


    def kill_simulation(self):
        self.state_file.close()
        self.plot_file.close()
        if os.path.isdir(self.parm.folder_name): shutil.rmtree(self.parm.folder_name)
        print('simulation terminated - folders and files deleted')
        sys.exit()


if __name__ == '__main__':      ###########################################    MAIN PROGRAM     ###################################################

    print('\n****************************************  Initializing Simulation  **************************************************\n')
    net = Network()                                                                                     # initiate the simulation
    print('\n******************************************  Beginning Simulation  ***************************************************\n')
    net.run_network()                                                                                               # runs the simulation
    if net.parm.print_records: net.print_network()                                                                  # prompts user to print output to the terminal
    if any([net.parm.show_video, net.parm.save_state_video, net.parm.save_change_video, net.parm.save_images]):     # calls rendering routine for video and images
        rd.display(net.parm.folder_name, net.state_filename, net.parm.show_video, net.parm.save_state_video, net.parm.save_change_video, net.parm.save_images)
    if net.parm.save_plots: rd.makeplots(net.parm.folder_name, net.plot_filename)                                   # plots various average simulation values vs time
