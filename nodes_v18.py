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
import math


class MakeNode(object):
    '''
    Class for creating node instances by type
    '''
    def Factory(node_id, node_class, node_states, node_polarity, node_period, records, energy_factor):
        '''
        Factory for initiating nodes
        '''
        if node_class == 'discrete':         return Node(node_id, node_class, node_states, node_polarity, node_period, records, energy_factor)
        if node_class == 'bias':             return Bias(node_id, node_class, node_states, node_polarity, node_period, records, energy_factor)

        assert False, 'Bad node creation: ' + node_class

    Factory = staticmethod(Factory)


class Node(object):
    '''
    Generic Node Class with methods used by all nodes
    '''

    def __init__(self, node_id, node_class, node_states, node_polarity, node_period, records, energy_factor):
        '''
        :param node_id: Unique number identifying the node
        :param node_type: A string specifying the type of node
        :param records: Number of time steps to retain in network, node and synapse histories
        :return: no return value
        '''
        self.node_id = node_id
        self.node_class = node_class
        self.node_states = node_states
        self.polarity = node_polarity
        self.period = node_period
        self.records = records
        self.energy_factor = energy_factor
        self.energy_factor_4x = 4.0 * energy_factor

#       Initialize node state variables
        self.connections = 0
        self.solve = False

#       Initialize node histories
        self.history = []
        self.solution_history = []

#       Initialize synapse data structures
        self.synapse_list = []
        self.external_synapse_list = []
        self.voltage = {}
        self.charge = {}
        self.weight = {}
        self.target = {}
        self.update_synapse_state = {}
        self.push_synapse_state = {}
        self.pcpw = {}
        self.pcnw = {}
        self.ncpw = {}
        self.ncnw = {}
        self.pvpw = {}
        self.pvnw = {}
        self.nvpw = {}
        self.nvnw = {}
        self.pcpw_input_list = []
        self.pcnw_input_list = []
        self.ncpw_input_list = []
        self.ncnw_input_list = []
        self.pwpc_list = []
        self.pwnc_list = []
        self.nwpc_list = []
        self.nwnc_list = []

#       Initialize node type specific variables and data structures
        if self.node_class == 'discrete':
            if self.node_states == 2:
                self.node_type = 'binary'
            if self.node_states == 3:
                self.node_type = 'ternary'
            if self.node_states > 3:
                self.node_type = 'x-nary'

        if self.node_class == 'sat':
            if self.node_states == 2:
                self.node_type = 'binary_sat'
            if self.node_states == 3:
                self.node_type = 'ternary_sat'

        self.entropy_factor =  np.log(self.node_states)
        self.state_array = -1.0 + 2.0 * np.array(list(range(self.node_states))) / (self.node_states - 1)
        self.state = np.random.choice(list(self.state_array))
        self.state_last = np.random.choice(list(self.state_array))
        self.neg_focus_array = (abs(self.state_array) - self.state_array)/2.0
        self.pos_focus_array = (abs(self.state_array) + self.state_array)/2.0
##        self.neg_focus_array = (1.0 - self.state_array)/2.0
##        self.pos_focus_array = (1.0 + self.state_array)/2.0
        self.neg_focus_energy_array = self.energy_factor_4x * self.neg_focus_array**2 * self.state_array
        self.pos_focus_energy_array = self.energy_factor_4x * self.pos_focus_array**2 * self.state_array


    def add_synapse(self, synapse_id, node_id, weight_type, weight_target, update_state_callback, push_state_callback):
        '''
        Adds synapses and initializes data structure for that synapse.
        Called by the network object when adding a synapse to the node.
        '''
        self.connections += 1
        self.synapse_list.append(synapse_id)
        if node_id != self.node_id: self.external_synapse_list.append(synapse_id)
        self.target[synapse_id] = weight_target
        self.pcpw[synapse_id] = 0.0
        self.pcnw[synapse_id] = 0.0
        self.ncpw[synapse_id] = 0.0
        self.ncnw[synapse_id] = 0.0
        self.pvpw[synapse_id] = 0.0
        self.pvnw[synapse_id] = 0.0
        self.nvpw[synapse_id] = 0.0
        self.nvnw[synapse_id] = 0.0
        self.update_synapse_state[synapse_id] = update_state_callback
        self.push_synapse_state[synapse_id] = push_state_callback
        self.update_synapse_state[synapse_id](self.node_id, self.state, 0.0, 100000.0)


    def receive_context(self, synapse_id, voltage, weight):
        '''
        Receives input from synapses.  Called by the synapse object when it updates state.
        '''
        self.weight[synapse_id] = weight
        self.voltage[synapse_id] = voltage
        self.charge[synapse_id] = voltage * weight

        if synapse_id in self.pcpw_input_list: self.pcpw_input_list.remove(synapse_id)
        if synapse_id in self.pcnw_input_list: self.pcnw_input_list.remove(synapse_id)
        if synapse_id in self.ncpw_input_list: self.ncpw_input_list.remove(synapse_id)
        if synapse_id in self.ncnw_input_list: self.ncnw_input_list.remove(synapse_id)

        if self.weight[synapse_id] >= 0.0 and self.voltage[synapse_id] > 0.0: self.pcpw_input_list.append(synapse_id)
        if self.weight[synapse_id] >= 0.0 and self.voltage[synapse_id] < 0.0: self.ncpw_input_list.append(synapse_id)
        if self.weight[synapse_id] <= 0.0 and self.voltage[synapse_id] < 0.0: self.pcnw_input_list.append(synapse_id)
        if self.weight[synapse_id] <= 0.0 and self.voltage[synapse_id] > 0.0: self.ncnw_input_list.append(synapse_id)


    def sample_state(self):
        '''
        Creates a sample of the node state given the current context of aggregates input charge.
        '''
#       compute boltzmann distribution
        neg_state_energy = self.pcpw_sum * self.nwnc_sum + self.ncnw_sum * self.pwpc_sum
        pos_state_energy = self.pcnw_sum * self.pwnc_sum + self.ncpw_sum * self.nwpc_sum
        energy = -neg_state_energy * self.neg_focus_energy_array - pos_state_energy * self.pos_focus_energy_array
        emin = np.min(energy)
        probability = np.exp((emin - energy))
        Zp = np.sum(probability)
        probability /= Zp

#       sample a state from the distribution
        seed = np.random.random()
        j = 0
        while seed > probability[j]:
            seed -= probability[j]
            j += 1
        self.state = self.state_array[j]
        self.energy = energy[j] + self.energy_factor_4x * (-neg_state_energy + pos_state_energy)
        self.neg_focus = self.neg_focus_array[j]
        self.pos_focus = self.pos_focus_array[j]

#       compute node statistics
        free_energy = emin - np.log(Zp)
        surprise_array = (energy - free_energy)
        self.entropy = np.sum(probability * surprise_array) / self.entropy_factor


    def relax_state(self):
        '''
        Reversibly updates charge context, generates node state sample and updates synapses.  Used in Gibbs sampling of the network node states.
        '''
#       create temporary state variables
        pcpw = cp.copy(self.pcpw)
        pcnw = cp.copy(self.pcnw)
        ncpw = cp.copy(self.ncpw)
        ncnw = cp.copy(self.ncnw)

#       update temporary state variables
        for i in self.synapse_list:
            if self.weight[i] >= 0.0: pcnw[i] = ncnw[i] = 0.0               # clears state variables for weight sign changes
            if self.weight[i] <= 0.0: pcpw[i] = ncpw[i] = 0.0               # clears state variables for weight sign changes

        for i in self.pcpw_input_list: pcpw[i] = min(pcpw[i] + self.charge[i], self.weight[i])
        for i in self.pcnw_input_list: pcnw[i] = min(pcnw[i] + self.charge[i], -self.weight[i])
        for i in self.ncpw_input_list: ncpw[i] = max(ncpw[i] + self.charge[i], -self.weight[i])
        for i in self.ncnw_input_list: ncnw[i] = max(ncnw[i] + self.charge[i], self.weight[i])

        self.pcpw_sum = sum([pcpw[i] for i in self.synapse_list if pcpw[i] > 0.0])
        self.pcnw_sum = sum([pcnw[i] for i in self.synapse_list if pcnw[i] > 0.0])
        self.ncpw_sum = sum([ncpw[i] for i in self.synapse_list if ncpw[i] < 0.0])
        self.ncnw_sum = sum([ncnw[i] for i in self.synapse_list if ncnw[i] < 0.0])

        self.pwpc_sum = sum([self.weight[i] for i in self.synapse_list if pcpw[i] > 0.0])
        self.nwpc_sum = sum([self.weight[i] for i in self.synapse_list if pcnw[i] > 0.0])
        self.pwnc_sum = sum([self.weight[i] for i in self.synapse_list if ncpw[i] < 0.0])
        self.nwnc_sum = sum([self.weight[i] for i in self.synapse_list if ncnw[i] < 0.0])

#       sample state and update the network
        self.sample_state()
        self.state_change = (self.state - self.state_last)/2.0
##        for i in self.external_synapse_list: self.push_synapse_state[i](self.node_id, self.state)
        for i in self.synapse_list: self.push_synapse_state[i](self.node_id, self.state)


    def update_state(self, weight_update):
        '''
        Irreversibly updates charge context, generates node state sample, generates weight error updates and updates synapses.  Used after the Gibbs sampling of network node states
        '''
#       update state variables
        for i in self.synapse_list:
            if self.weight[i] >= 0.0: self.pcnw[i] = self.ncnw[i] = self.nvnw[i] = self.pvnw[i] = 0.0               # clears state variables for weight sign changes
            if self.weight[i] <= 0.0: self.pcpw[i] = self.ncpw[i] = self.nvpw[i] = self.pvpw[i] = 0.0               # clears state variables for weight sign changes

        for i in self.pcpw_input_list:
            self.pcpw[i] = min(self.pcpw[i] + self.charge[i], self.weight[i])
            self.pvpw[i] = min(self.pvpw[i] + self.voltage[i], 1.0)
        for i in self.pcnw_input_list:
            self.pcnw[i] = min(self.pcnw[i] + self.charge[i], -self.weight[i])
            self.nvnw[i] = max(self.nvnw[i] + self.voltage[i], -1.0)
        for i in self.ncpw_input_list:
            self.ncpw[i] = max(self.ncpw[i] + self.charge[i], -self.weight[i])
            self.nvpw[i] = max(self.nvpw[i] + self.voltage[i], -1.0)
        for i in self.ncnw_input_list:
            self.ncnw[i] = max(self.ncnw[i] + self.charge[i], self.weight[i])
            self.pvnw[i] = min(self.pvnw[i] + self.voltage[i], 1.0)

        pcpw_list = [i for i in self.synapse_list if self.pcpw[i] > 0.0]
        pcnw_list = [i for i in self.synapse_list if self.pcnw[i] > 0.0]
        ncpw_list = [i for i in self.synapse_list if self.ncpw[i] < 0.0]
        ncnw_list = [i for i in self.synapse_list if self.ncnw[i] < 0.0]
        zc_list = [i for i in self.synapse_list if i not in (pcpw_list + pcnw_list + ncpw_list + ncnw_list)]

        self.pcpw_sum = sum([self.pcpw[i] for i in pcpw_list])
        self.pcnw_sum = sum([self.pcnw[i] for i in pcnw_list])
        self.ncpw_sum = sum([self.ncpw[i] for i in ncpw_list])
        self.ncnw_sum = sum([self.ncnw[i] for i in ncnw_list])

        self.pwpc_sum = sum([self.weight[i] for i in pcpw_list])
        self.nwpc_sum = sum([self.weight[i] for i in pcnw_list])
        self.pwnc_sum = sum([self.weight[i] for i in ncpw_list])
        self.nwnc_sum = sum([self.weight[i] for i in ncnw_list])

        pvpw_sum = sum([self.pvpw[i] for i in pcpw_list])
        nvnw_sum = sum([self.nvnw[i] for i in pcnw_list])
        nvpw_sum = sum([self.nvpw[i] for i in ncpw_list])
        pvnw_sum = sum([self.pvnw[i] for i in ncnw_list])

#       sample state
        self.sample_state()

#       update history
        self.state_change = (self.state - self.state_last)/2.0
        self.state_last = cp.copy(self.state)
        self.history.append((self.node_id, self.node_type, self.connections, self.state, self.energy, self.entropy))
        if len(self.history) > self.records: self.history.pop(0)

#       compute weight errors
        f2 = self.neg_focus
        f1 = self.pos_focus
        f2c = 1.0 - f2
        f1c = 1.0 - f1
        if weight_update:
            state_2 = self.state**2
            state_2_sign = state_2 * np.sign(self.state)
            w2_avg = sum([self.weight[i]**2 for i in self.synapse_list]) / self.connections

            denom1 = pvpw_sum + len(ncnw_list) * abs(self.state)
            denom2 = pvnw_sum + len(pcpw_list) * abs(self.state)
            if denom1 >0 and denom2 > 0.0:
                error1 = -f2 * (self.pcpw_sum - self.nwnc_sum * self.state) / denom1
                error2 = -f2 * (self.ncnw_sum - self.pwpc_sum * self.state) / denom2
                for i in pcpw_list:
                    weight_error = (error1 * self.pvpw[i]**2 - error2 * state_2_sign) / (self.pvpw[i]**2 + state_2)
                    self.update_synapse_state[i](self.node_id, self.state, weight_error, w2_avg)
                    self.pcpw[i] *= f2c
                    self.pvpw[i] *= f2c
                for i in ncnw_list:
                    weight_error = (error2 * self.pvnw[i]**2 - error1 * state_2_sign) / (self.pvnw[i]**2 + state_2)
                    self.update_synapse_state[i](self.node_id, self.state, weight_error, w2_avg)
                    self.ncnw[i] *= f2c
                    self.pvnw[i] *= f2c
            else:
                for i in pcpw_list + ncnw_list: self.push_synapse_state[i](self.node_id, self.state)

            denom1 = -nvnw_sum + len(ncpw_list) * abs(self.state)
            denom2 = -nvpw_sum + len(pcnw_list) * abs(self.state)
            if denom1 > 0.0 and denom2 > 0.0:
                error1 = -f1 * (self.pcnw_sum - self.pwnc_sum * self.state) / denom1
                error2 = -f1 * (self.ncpw_sum - self.nwpc_sum * self.state) / denom2
                for i in pcnw_list:
                    weight_error = (-error1 * self.nvnw[i]**2 - error2 * state_2_sign) / (self.nvnw[i]**2 + state_2)
                    self.update_synapse_state[i](self.node_id, self.state, weight_error, w2_avg)
                    self.pcnw[i] *= f1c
                    self.nvnw[i] *= f1c
                for i in ncpw_list:
                    weight_error = (-error2 * self.nvpw[i]**2 - error1 * state_2_sign) / (self.nvpw[i]**2 + state_2)
                    self.update_synapse_state[i](self.node_id, self.state, weight_error, w2_avg)
                    self.ncpw[i] *= f1c
                    self.nvpw[i] *= f1c
            else:
                for i in pcnw_list + ncpw_list: self.push_synapse_state[i](self.node_id, self.state)

            for i in zc_list:
                self.push_synapse_state[i](self.node_id, self.state)

        else:
            for i in pcpw_list:
                self.push_synapse_state[i](self.node_id, self.state)
                self.pcpw[i] *= f2c
                self.pvpw[i] *= f2c
            for i in ncnw_list:
                self.push_synapse_state[i](self.node_id, self.state)
                self.ncnw[i] *= f2c
                self.pvnw[i] *= f2c
            for i in pcnw_list:
                self.push_synapse_state[i](self.node_id, self.state)
                self.pcnw[i] *= f1c
                self.nvnw[i] *= f1c
            for i in ncpw_list:
                self.push_synapse_state[i](self.node_id, self.state)
                self.ncpw[i] *= f1c
                self.nvpw[i] *= f1c

#       compute dissipation, transport and quality factors for accumalation by the network object collecting statistics on the network
        self.dissipation =  (f2 * (self.pcpw_sum - self.nwnc_sum * self.state))**2 + (f2 * (self.ncnw_sum - self.pwpc_sum * self.state))**2
        self.dissipation += (f1 * (self.pcnw_sum - self.pwnc_sum * self.state))**2 + (f1 * (self.ncpw_sum - self.nwpc_sum * self.state))**2
        self.transport =  (f2 * (self.pcpw_sum + self.nwnc_sum * self.state))**2 + (f2 * (self.ncnw_sum + self.pwpc_sum * self.state))**2
        self.transport += (f1 * (self.pcnw_sum + self.pwnc_sum * self.state))**2 + (f1 * (self.ncpw_sum + self.nwpc_sum * self.state))**2
        self.transport /= 4
        self.quality_denom =  f2 * (abs(self.pcpw_sum - self.nwnc_sum * self.state) + abs(self.ncnw_sum - self.pwpc_sum * self.state))
        self.quality_denom += f1 * (abs(self.pcnw_sum - self.pwnc_sum * self.state) + abs(self.ncpw_sum - self.nwpc_sum * self.state))
        self.quality_numer =  f2 * (abs(self.pcpw_sum + self.nwnc_sum * self.state) + abs(self.ncnw_sum + self.pwpc_sum * self.state))
        self.quality_numer += f1 * (abs(self.pcnw_sum + self.pwnc_sum * self.state) + abs(self.ncpw_sum + self.nwpc_sum * self.state))


class Bias(Node):   #       *************************     BIAS Node Class     **************************************
    '''
    BIAS node class to create potentials and charge to inject into the network.
    '''
    def __init__(self, node_id, node_class, node_states, node_polarity, node_period, records, energy_factor):
        Node.__init__(self, node_id, node_class, node_states, node_polarity, node_period, records, energy_factor)
        self.node_type = 'bias'
        self.state = self.polarity
        self.state_change = 0.0

    def receive_context(self, synapse_id, voltage, weight):
        self.weight[synapse_id] = weight
        self.voltage[synapse_id] = voltage
        self.charge[synapse_id] = voltage * weight

    def set_state(self, time, era, logic_mode):
        if logic_mode == 'driven':
            period = self.period[min(len(self.period)-1, era)]
            if 2 * ((time-1) % period) < period: self.state = self.polarity
            else: self.state = -self.polarity
        if logic_mode == 'off': self.state = 0.0
        if logic_mode == 'noise': self.state = np.random.choice([-1.0,1.0])
        if logic_mode == 'reflect':
            energy = sum([(self.charge[i] + self.state_array * self.weight[i])**2 for i in self.synapse_list])
            emin = np.min(energy)
            probability = np.exp((emin - energy))
            Zp = np.sum(probability)
            probability /= Zp
            seed = np.random.random()
            j = 0
            while seed > probability[j]:
                seed -= probability[j]
                j += 1
            self.state = self.state_array[j]

    def relax_state(self, predict_mode):
        if predict_mode:
            for i in self.synapse_list:
                if self.state * self.voltage[i] < 0.0:
                    self.push_synapse_state[i](self.node_id, self.state)
                else:
                    self.push_synapse_state[i](self.node_id, 0.0)
        else:
            for i in self.synapse_list:
                self.push_synapse_state[i](self.node_id, self.state)

    def update_state(self, predict_mode, weight_update):
        if predict_mode:
            if weight_update:
                for i in self.synapse_list:
                    if self.state * self.voltage[i] < 0.0:
                        w2_avg = sum([self.weight[i]**2 for i in self.synapse_list]) / self.connections
                        self.update_synapse_state[i](self.node_id, self.state, self.target[i] - self.weight[i], w2_avg)
                    else:
                        self.push_synapse_state[i](self.node_id, 0.0)
            else:
               for i in self.synapse_list:
                    if self.state * self.voltage[i] < 0.0:
                        self.push_synapse_state[i](self.node_id, self.state)
                    else:
                        self.push_synapse_state[i](self.node_id, 0.0)
        else:
            if weight_update:
                w2_avg = sum([self.weight[i]**2 for i in self.synapse_list]) / self.connections
                for i in self.synapse_list: self.update_synapse_state[i](self.node_id, self.state, self.target[i] - self.weight[i], w2_avg)
            else:
                for i in self.synapse_list: self.push_synapse_state[i](self.node_id, self.state)

    def evaluate_state(self):
#       compute thermodynamics
        self.energy = sum([(self.charge[i] + self.state * self.weight[i])**2 for i in self.synapse_list])
        self.solve = all([(self.state * self.voltage[i] < 0.0) for i in self.synapse_list])
        self.dissipation = 0.0
        self.transport = 0.0
        self.quality_denom = 0.0
        self.quality_numer = 0.0
        self.entropy = 0.0

#       update history
        self.history.append((self.node_id, self.node_type, self.connections, self.state, self.energy, self.entropy))
        if len(self.history) > self.records: self.history.pop(0)
        self.solution_history.append(self.solve)
        if len(self.solution_history) > self.records: self.solution_history.pop(0)
