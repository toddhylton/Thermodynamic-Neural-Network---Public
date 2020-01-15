import numpy as np
import copy as cp
import math


class MakeNode(object):
    '''
    Class for creating node instances by type
    '''
    def Factory(node_id, node_class, node_states, node_polarity, node_period, records, energy_factor, threshold):
        '''
        Factory for initiating nodes
        '''
        if node_class == 'discrete':         return Node(node_id, node_class, node_states, node_polarity, node_period, records, energy_factor, threshold)
        if node_class == 'bias':             return Bias(node_id, node_class, node_states, node_polarity, node_period, records, energy_factor, threshold)

        assert False, 'Bad node creation: ' + node_class

    Factory = staticmethod(Factory)


class Node(object):
    '''
    Generic Node Class with methods used by all nodes
    '''

    def __init__(self, node_id, node_class, node_states, node_polarity, node_period, records, energy_factor, threshold):
        '''
        :param node_id: Unique number identifying the node
        :param node_type: A string specifying the type of node
        :param records: Number of time steps to retain in network, node and synapse histories
        :return: no return value
        '''
#       Initialize node state variables
        self.node_id = node_id
        self.node_class = node_class
        self.node_states = node_states
        self.polarity = node_polarity
        self.period = node_period
        self.records = records
        self.energy_factor = energy_factor
        self.threshold = threshold
        self.energy_factor = energy_factor
        self.energy_factor_4x = 4.0 * energy_factor
        self.fluctuation = True 
        self.connections = 0
        self.solve = False
        self.state_last = 0.0
        self.energy_last = 0.0
        self.dissipation = 1.0
        self.transport = 1.0
        self.quality_denom = 1.0
        self.quality_numer = 1.0
        if self.node_class == 'discrete':
            if self.node_states == 2:
                self.node_type = 'binary'
            if self.node_states == 3:
                self.node_type = 'ternary'
            if self.node_states > 3:
                self.node_type = 'x-nary'

#       Initialize node history data structures
        self.history = []

#       Initialize edge data structures
        self.synapse_list = []
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
        self.pwpc = {}
        self.nwpc = {}
        self.pwnc = {}
        self.nwnc = {}


    def add_synapse(self, synapse_id, weight_target, update_state_callback, push_state_callback):
        '''
        Adds synapses and initializes data structure for that synapse.
        Called by the network object when adding a synapse to the node.
        '''
        self.connections += 1
        self.synapse_list.append(synapse_id)
        self.target[synapse_id] = weight_target
        self.weight[synapse_id] = 0.0
        self.voltage[synapse_id] = 0.0
        self.charge[synapse_id] = 0.0
        self.pcpw[synapse_id] = self.pcnw[synapse_id] = self.ncpw[synapse_id] = self.ncnw[synapse_id] = 0.0
        self.pvpw[synapse_id] = self.pvnw[synapse_id] = self.nvpw[synapse_id] = self.nvnw[synapse_id] = 0.0
        self.pwpc[synapse_id] = self.nwpc[synapse_id] = self.pwnc[synapse_id] = self.nwnc[synapse_id] = 0.0
        self.update_synapse_state[synapse_id] = update_state_callback
        self.push_synapse_state[synapse_id] = push_state_callback
        

    def receive_context(self, synapse_id, voltage, weight):
        '''
        Receives input from edges, stores input states, and updates compartment lists.
        '''        
        self.weight[synapse_id] = weight
        self.voltage[synapse_id] = voltage
        self.charge[synapse_id] = voltage * weight


    def sample_state(self):
        '''
        Creates a sample of the node state given the current context (node, edge and compartment states plus recent inputs).
        ''' 
#       select distribution of states by filtering unlikely / high energy states and set the separation between states using the node temperature scale (self.threshold)

        neg_state_energy = self.energy_factor_4x * (self.pcpw_sum * self.nwnc_sum + self.ncnw_sum * self.pwpc_sum)
        pos_state_energy = self.energy_factor_4x * (self.pcnw_sum * self.pwnc_sum + self.ncpw_sum * self.nwpc_sum)
        neg_states_max = max(1, int(-neg_state_energy / self.threshold))
        pos_states_max = max(1, int(pos_state_energy / self.threshold))
        total_states = max(3, int((pos_state_energy - neg_state_energy + 1) / self.threshold))
        if neg_states_max > pos_states_max:
            neg_states_min = max(0, neg_states_max - self.node_states)
            pos_states_min = min(neg_states_min, pos_states_max)
        else:
            pos_states_min = max(0, pos_states_max - self.node_states)
            neg_states_min = min(pos_states_min, neg_states_max)
        neg_state_array = np.arange(-neg_states_max, -neg_states_min) / neg_states_max
        pos_state_array = np.arange(pos_states_min+1, pos_states_max+1) / pos_states_max
        if pos_states_min == 0 or neg_states_min == 0: zero_array = np.array([0])
        else: zero_array = np.array([])        
        state_array = np.concatenate([neg_state_array, zero_array, pos_state_array])

#       compute state energies and probability distribution using a boltzmann distribution
        energy = (-neg_state_energy * (state_array < 0.0) - pos_state_energy * (state_array > 0.0)) * state_array
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
        self.state = state_array[j]

#       evaluate state changes
        self.state_change = (self.state - self.state_last)/2.0
        self.state_last = cp.copy(self.state)        
        self.energy = energy[j]
        self.fluctuation = (abs(energy[j] - self.energy_last) > self.threshold)
        self.energy_last = energy[j]

#       compute node statistics
        self.free_energy = emin - np.log(Zp)
        surprise_array = (energy - self.free_energy)
        self.entropy = np.sum(probability * surprise_array) / np.log(total_states)

#       update node history        
        self.history.append((self.node_id, self.node_type, self.connections, self.state, self.energy, self.entropy))
        if len(self.history) > self.records: self.history.pop(0)


    def update_state(self, weight_update):
        '''
        Updates node, edge weight, edge charge and compartment charge states.
        '''
#       create (temporary) edge state variables
        pcpw = cp.copy(self.pcpw)
        pcnw = cp.copy(self.pcnw)
        ncpw = cp.copy(self.ncpw)
        ncnw = cp.copy(self.ncnw)

        pwpc = cp.copy(self.pwpc)
        nwpc = cp.copy(self.nwpc)
        pwnc = cp.copy(self.pwnc)
        nwnc = cp.copy(self.nwnc)
        
#       reversibly update edge charges
        pcpw_input_list = [i for i in self.synapse_list if self.charge[i] > 0.0 and self.weight[i] > 0.0]
        pcnw_input_list = [i for i in self.synapse_list if self.charge[i] > 0.0 and self.weight[i] < 0.0]
        ncpw_input_list = [i for i in self.synapse_list if self.charge[i] < 0.0 and self.weight[i] > 0.0]
        ncnw_input_list = [i for i in self.synapse_list if self.charge[i] < 0.0 and self.weight[i] < 0.0]
        
        for i in pcpw_input_list: pcpw[i] = min(pcpw[i] + self.charge[i], self.weight[i])
        for i in pcnw_input_list: pcnw[i] = min(pcnw[i] + self.charge[i], -self.weight[i])
        for i in ncpw_input_list: ncpw[i] = max(ncpw[i] + self.charge[i], -self.weight[i])
        for i in ncnw_input_list: ncnw[i] = max(ncnw[i] + self.charge[i], self.weight[i])

        for i in pcpw_input_list: pwpc[i] = self.weight[i]
        for i in pcnw_input_list: nwpc[i] = self.weight[i]
        for i in ncpw_input_list: pwnc[i] = self.weight[i]
        for i in ncnw_input_list: nwnc[i] = self.weight[i]
            
#       compute compartment charges and weights
        pcpw_list = [i for i in self.synapse_list if pcpw[i] > 0.0]
        pcnw_list = [i for i in self.synapse_list if pcnw[i] > 0.0]
        ncpw_list = [i for i in self.synapse_list if ncpw[i] < 0.0]
        ncnw_list = [i for i in self.synapse_list if ncnw[i] < 0.0]
        zc_list = [i for i in self.synapse_list if i not in (pcpw_list + pcnw_list + ncpw_list + ncnw_list)]
        
        self.pcpw_sum = sum([pcpw[i] for i in pcpw_list])
        self.pcnw_sum = sum([pcnw[i] for i in pcnw_list])
        self.ncpw_sum = sum([ncpw[i] for i in ncpw_list])
        self.ncnw_sum = sum([ncnw[i] for i in ncnw_list])

        self.pwpc_sum = sum([pwpc[i] for i in pcpw_list])
        self.nwpc_sum = sum([nwpc[i] for i in pcnw_list])
        self.pwnc_sum = sum([pwnc[i] for i in ncpw_list])
        self.nwnc_sum = sum([nwnc[i] for i in ncnw_list])

#       sample node state
        self.sample_state()

#       Fluctuate => Make a reversible update to the edges state variables when there is a thermally significant fluctuation in node energy
        if self.fluctuation or not weight_update or self.state == 0.0:      
            for i in self.synapse_list: self.push_synapse_state[i](self.node_id, self.state)

#       Equilibrate => Make irreversible updates to the edge state variables when there is a thermally insignificant fluctuation in node energy 
        else:
       
#           Irreversibly update edge charges, weights and voltages 
            self.pcpw = cp.copy(pcpw)
            self.pcnw = cp.copy(pcnw)
            self.ncpw = cp.copy(ncpw)
            self.ncnw = cp.copy(ncnw)

            self.pwpc = cp.copy(pwpc)
            self.nwpc = cp.copy(nwpc)
            self.pwnc = cp.copy(pwnc)
            self.nwnc = cp.copy(nwnc)

            for i in pcpw_input_list: self.pvpw[i] = min(self.pvpw[i] + self.voltage[i], 1.0)
            for i in pcnw_input_list: self.nvnw[i] = max(self.nvnw[i] + self.voltage[i], -1.0)
            for i in ncpw_input_list: self.nvpw[i] = max(self.nvpw[i] + self.voltage[i], -1.0)
            for i in ncnw_input_list: self.pvnw[i] = min(self.pvnw[i] + self.voltage[i], 1.0)

#           compute compartment voltages
            pvpw_sum = sum([self.pvpw[i] for i in pcpw_list])
            nvnw_sum = sum([self.nvnw[i] for i in pcnw_list])
            nvpw_sum = sum([self.nvpw[i] for i in ncpw_list])
            pvnw_sum = sum([self.pvnw[i] for i in ncnw_list])

#           update edge weights and charges, update the network
            if self.state < 0.0:
                state_2 = self.state**2
                denom1 = pvpw_sum + len(ncnw_list) * abs(self.state)
                denom2 = pvnw_sum + len(pcpw_list) * abs(self.state)
                if denom1 > 0.0 and denom2 > 0.0:
##                    w2_avg = sum([self.weight[i]**2 for i in self.synapse_list]) / (len(pcpw_list) + len(ncnw_list))
                    w2_avg = sum([self.weight[i]**2 for i in (pcpw_list + ncnw_list)]) / (len(pcpw_list) + len(ncnw_list))
                    error1 = -(self.pcpw_sum - self.nwnc_sum * self.state) / denom1 / 2.0                                #the 1/2 is because each node contributes 1/2 of the error update
                    error2 = -(self.ncnw_sum - self.pwpc_sum * self.state) / denom2 / 2.0                                #the 1/2 is because each node contributes 1/2 of the error update
                    for i in pcpw_list:
                        weight_error = (error1 * self.pvpw[i]**2 + error2 * state_2) / (self.pvpw[i]**2 + state_2)
                        self.update_synapse_state[i](self.node_id, self.state, weight_error, w2_avg)               
                        self.pcpw[i] = self.pwpc[i] = self.pvpw[i] = 0.0
                    for i in ncnw_list:
                        weight_error = (error2 * self.pvnw[i]**2 + error1 * state_2) / (self.pvnw[i]**2 + state_2)
                        self.update_synapse_state[i](self.node_id, self.state, weight_error, w2_avg)
                        self.ncnw[i] = self.nwnc[i] = self.pvnw[i] = 0.0
                else:
                    for i in pcpw_list + ncnw_list: self.push_synapse_state[i](self.node_id, self.state)
                for i in pcnw_list + ncpw_list + zc_list: self.push_synapse_state[i](self.node_id, self.state)
                self.dissipation =  self.energy_factor * ((self.pcpw_sum - self.nwnc_sum * self.state)**2 + (self.ncnw_sum - self.pwpc_sum * self.state)**2)
                self.transport =  self.energy_factor * ((self.pcpw_sum + self.nwnc_sum * self.state)**2 + (self.ncnw_sum + self.pwpc_sum * self.state)**2) / 4
                self.quality_denom = abs(self.pcpw_sum - self.nwnc_sum * self.state) + abs(self.ncnw_sum - self.pwpc_sum * self.state)
                self.quality_numer = abs(self.pcpw_sum + self.nwnc_sum * self.state) + abs(self.ncnw_sum + self.pwpc_sum * self.state)
                    
            if self.state > 0.0:
                state_2 = self.state**2
                denom1 = -nvnw_sum + len(ncpw_list) * abs(self.state)
                denom2 = -nvpw_sum + len(pcnw_list) * abs(self.state)
                if denom1 > 0.0 and denom2 > 0.0:
##                    w2_avg = sum([self.weight[i]**2 for i in self.synapse_list]) / (len(pcnw_list) + len(ncpw_list))
                    w2_avg = sum([self.weight[i]**2 for i in (pcnw_list + ncpw_list)]) / (len(pcnw_list) + len(ncpw_list))
                    error1 = -(self.pcnw_sum - self.pwnc_sum * self.state) / denom1 / 2.0                                #the 1/2 is because each node contributes 1/2 of the error update
                    error2 = -(self.ncpw_sum - self.nwpc_sum * self.state) / denom2 / 2.0                                #the 1/2 is because each node contributes 1/2 of the error update
                    for i in pcnw_list:
                        weight_error = (-error1 * self.nvnw[i]**2 - error2 * state_2) / (self.nvnw[i]**2 + state_2)
                        self.update_synapse_state[i](self.node_id, self.state, weight_error, w2_avg)
                        self.pcnw[i] = self.nwpc[i] = self.nvnw[i] = 0.0
                    for i in ncpw_list:
                        weight_error = (-error2 * self.nvpw[i]**2 - error1 * state_2) / (self.nvpw[i]**2 + state_2)
                        self.update_synapse_state[i](self.node_id, self.state, weight_error, w2_avg)
                        self.ncpw[i] = self.pwnc[i] = self.nvpw[i] = 0.0                                                ############################################ fixed error
                else:
                    for i in pcnw_list + ncpw_list: self.push_synapse_state[i](self.node_id, self.state)
                for i in pcpw_list + ncnw_list + zc_list: self.push_synapse_state[i](self.node_id, self.state)
                self.dissipation = self.energy_factor * ((self.pcnw_sum - self.pwnc_sum * self.state)**2 + (self.ncpw_sum - self.nwpc_sum * self.state)**2)
                self.transport = self.energy_factor * ((self.pcnw_sum + self.pwnc_sum * self.state)**2 + (self.ncpw_sum + self.nwpc_sum * self.state)**2) / 4
                self.quality_denom = abs(self.pcnw_sum - self.pwnc_sum * self.state) + abs(self.ncpw_sum - self.nwpc_sum * self.state)
                self.quality_numer = abs(self.pcnw_sum + self.pwnc_sum * self.state) + abs(self.ncpw_sum + self.nwpc_sum * self.state)
                

class Bias(Node):   #       *************************     BIAS Node Class     **************************************
    '''
    BIAS node class to create potentials and charge to inject into the network.
    '''    
    def __init__(self, node_id, node_class, node_states, node_polarity, node_period, records, energy_factor, threshold):
        Node.__init__(self, node_id, node_class, node_states, node_polarity, node_period, records, energy_factor, threshold)
        self.node_type = 'bias'
        self.state = self.polarity
        self.state_change = 0.0
        self.fluctuation = False
        self.solution_history = []

    def receive_context(self, synapse_id, voltage, weight):
        self.weight[synapse_id] = weight
        self.voltage[synapse_id] = voltage
        self.charge[synapse_id] = voltage * weight
        self.fluctuation = True

    def update_state(self, time, era, weight_update, mode): 
#       compute node state
        self.state_last = cp.copy(self.state)
        period = self.period[min(len(self.period)-1, era)]
        if 2 * ((time-1) % period) < period: self.state = self.polarity
        else: self.state = -self.polarity
        if mode == 'off': self.state = 0.0
        if mode == 'noise': self.state = np.random.choice([-1.0,1.0])
        if mode == 'reflect':
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
        if mode == 'predict':
            if any([self.state * self.voltage[i] > 0.0] for i in self.synapse_list): self.state = 0.0         

#       update edges
        if (self.state != self.state_last) or self.fluctuation:
            if weight_update:
                w2_avg = sum([self.weight[i]**2 for i in self.synapse_list]) / self.connections
                for i in self.synapse_list: self.update_synapse_state[i](self.node_id, self.state, self.target[i] - self.weight[i], w2_avg)
            else:
               for i in self.synapse_list: self.push_synapse_state[i](self.node_id, self.state)
            self.fluctuation = False


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
        
  










        


