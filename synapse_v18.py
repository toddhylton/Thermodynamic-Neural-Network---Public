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
import scipy.special as sps
import math
import sys


class MakeSynapse(object):   # *******************************  Make Synapse Object *********************************************
    '''
    Class for initiating synapses
    '''

    def Factory(synapse_id, weight_type, energy_factor, time_depth, weight_bound, weight_target, weight_noise, size_mass, change_mass, records):
        '''
        Factory for Synapse Object creation
        '''
        if weight_type == 'real1':  return Real1(synapse_id, energy_factor, time_depth, weight_bound, weight_target, weight_noise, size_mass, change_mass, records)
        if weight_type == 'real2':  return Real2(synapse_id, energy_factor, time_depth, weight_bound, weight_target, weight_noise, size_mass, change_mass, records)
        if weight_type == 'fixed':  return Fixed(synapse_id, energy_factor, time_depth, weight_bound, weight_target, weight_noise, size_mass, change_mass, records)

        assert False, "Bad synapse creation: " + weight_type

    Factory = staticmethod(Factory)


class Synapse(object):   # *******************************  Synapse Object *********************************************
    '''
    Generic synapse class implementing methods used by all synapse classes
    '''

    def __init__(self, synapse_id, energy_factor, time_depth, weight_bound, weight_target, weight_noise, size_mass, change_mass, records):
        '''
        Initiate a Synapse
        '''
        self.synapse_id = synapse_id
        self.bound_high = abs(weight_bound)
        self.bound_low = -self.bound_high
        self.time_depth = time_depth
        self.weight_target = weight_target
        self.weight_noise = weight_noise
        self.size_mass = size_mass
        self.change_mass = change_mass
        self.energy_factor = 2.0 * energy_factor  # 2.0 reflects node error formula (assuming strong decision states)
        self.prefactor = self.energy_factor + size_mass + change_mass
        self.records = records
        self.order = 0.0
        self.correlation = 0.0
        self.history = []


    def add_nodes(self, node_id_1, node_id_1_callback, node_id_2, node_id_2_callback):
        '''
        Adds nodes and initializes data structures for those nodes.
        Called by the network object when adding a nodes to a synapse.
        '''
        self.node_list = [node_id_1, node_id_2]
        self.node_pair = {node_id_1:node_id_2, node_id_2:node_id_1}
        self.send_context = {node_id_1:node_id_2_callback, node_id_2:node_id_1_callback}
        self.output_state = {node_id_1:0.0, node_id_2:0.0}
        self.input_queue = {node_id_1:self.time_depth*[0.0], node_id_2:self.time_depth*[0.0]}


    def push_state(self, node_id, input_state):
        '''
        Pushes node outputs through the synapse without weight update and delivers them to the connected node as input.
        Called by node objects when they relax or update state.
        '''
        self.input_queue[node_id].append(input_state)
        self.output_state[node_id] = self.input_queue[node_id].pop(0)
        self.send_context[node_id](self.synapse_id, self.output_state[node_id], self.weight)


    def update_state(self, node_id, input_state, input_weight_error, weight2_avg):
        '''
        Pushes node outputs through the synapse with weight update and delivers them to the connected node as input.
        Called by node objects when they update state.
        '''
        self.input_queue[node_id].append(input_state)
        self.output_state[node_id] = self.input_queue[node_id].pop(0)
        self.order = -input_state * self.output_state[self.node_pair[node_id]]
        self.weight_error = input_weight_error / 2.0                                #1/2 the error from each node
        self.delta = math.sqrt(1.0 + 1.0 / (2.0 * self.prefactor * weight2_avg))
        self.update_weight()
        self.history.append((self.synapse_id, self.weight_type, self.output_state[self.node_list[0]], self.output_state[self.node_list[1]], self.weight, self.prefactor, self.weight_error))
        if len(self.history) > self.records: self.history.pop(0)
        self.send_context[node_id](self.synapse_id, self.output_state[node_id], self.weight)


class Real1(Synapse):   ##################################    REAL1 Synapse Class   ############################################

    def __init__(self, synapse_id, energy_factor, time_depth, weight_bound, weight_target, weight_noise, size_mass, change_mass, records):
        Synapse.__init__(self, synapse_id, energy_factor, time_depth, weight_bound, weight_target, weight_noise, size_mass, change_mass, records)
        self.weight_type = 'real1'
        self.weight = np.random.randn() / np.math.sqrt(2.0 * self.prefactor)

    def update_weight(self):
        '''
        Function for executing the update of real valued synapses.
        Weight values are drawn from a gaussian but cutoff beyond [self.bound_low, self.bound_high]
        '''

        # compute weight update
        self.weight += (self.energy_factor * self.weight_error - self.size_mass * self.weight)/self.prefactor
        if self.weight_noise:
            self.weight /= self.delta
            self.weight += np.random.randn() / np.math.sqrt(2.0 * self.prefactor)
        self.weight = max(self.bound_low, min(self.bound_high, self.weight))


class Real2(Synapse):   ##################################    REAL2 Synapse Class   ############################################

    def __init__(self, synapse_id, energy_factor, time_depth, weight_bound, weight_target, weight_noise, size_mass, change_mass, records):
        Synapse.__init__(self, synapse_id, energy_factor, time_depth, weight_bound, weight_target, weight_noise, size_mass, change_mass, records)
        self.weight_type = 'real2'
        factor = np.math.sqrt(self.prefactor)
        R = np.random.random()
        self.weight = sps.erfcinv((1.0-R) * sps.erfc(factor * self.bound_low) + R * sps.erfc(factor * self.bound_high))

    def update_weight(self):
        '''
        Function for executing the update of real valued synapses.
        Weight values are drawn from a gaussian limited to the domain [self.bound_low, self.bound_high]
        '''

        # compute weight update
        self.weight += (self.energy_factor * self.weight_error  - self.size_mass * self.weight)/self.prefactor
        if self.weight_noise:
            self.weight /= self.delta
            factor = np.math.sqrt(self.prefactor)
            R = np.random.random()
            self.noise = sps.erfcinv((1.0-R) * sps.erfc(factor*(self.bound_low - self.weight)) + R * sps.erfc(factor*(self.bound_high - self.weight))) / factor
            self.weight += self.noise
        self.weight = max(self.bound_low, min(self.bound_high, self.weight))


class Fixed(Synapse):   ##################################    FIXED Synapse Class   ############################################

    def __init__(self, synapse_id, energy_factor, time_depth, weight_bound, weight_target, weight_noise, size_mass, change_mass, records):
        Synapse.__init__(self, synapse_id, energy_factor, time_depth, weight_bound, weight_target, weight_noise, size_mass, change_mass, records)
        self.weight_type = 'fixed'
        self.weight = self.weight_target

    def update_weight(self):
        '''
        Dummy update function for fixed weights
        '''
        return
