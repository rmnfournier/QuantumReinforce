from torch import optim
import torch
import Physics_Engine as PE
import Reinforce_Net as RN
import Reinforce_Hex_Net as RHN
torch.set_default_tensor_type('torch.DoubleTensor')
import numpy as np
import gmpy2
import random
from collections import deque
import pandas as pd
import time
from torch.distributions import chi2
import scipy.stats as stats
import os
import sys
dirname = os.path.dirname(__file__)



class QuantumReinforce_agent:
    def __init__(self, tau, lr, weight_decay,nb_samples,nb_warmup,architecture,lattice_size,u):
        '''
        Initialise the agent that learns according to the Quantum Reinforce algorithm
        :param tau: part of the network that is updated at each iteration
        :param lr: learning rate
        :param weight_decay: for stability
        :param nb_samples: number of samples for estimating the sums
        :param nb_warmup: number of samples before starting collecting states
        :param architecture:[list of integers] hidden layers
        :param lattice_size: size of the lattice
        '''
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device="cpu"
        self.u=u
        self.pe = PE.Physics_Engine(matrix_length=4, nb_flavors=2, filling=0.5,k=np.array([0,0]),U=u)

        #self.network = RN.Reinforce_Net(self.pe.nb_sites*2,1,architecture,device=self.device)
        self.network = RHN.Reinforce_Hex_Net(self.pe.nb_sites*2,1,architecture,device=self.device)
        self.network_optimizer = optim.Adadelta(self.network.parameters())
        self.target_network = self.network

        self.tau = tau
        self.lr = lr

        self.nb_samples=nb_samples
        self.nb_warmup = nb_warmup
        self.samples=[]
        self.samples_prob =[]

        self.list_loss=[] # allow to inspect the training

    def act(self, state):
        """ Compute the current estimation of H|state>
        @param state : state in occupation space
        @return a,b : where a = Re(H|state>) and b = Im(H|state>)
        """
        return self.network(state.unsqueeze(0))

    def prob(self, state):
        """ Compute the current estimation of |<state|psi>|^2
        @param state : vector in occupation representation
        @return p
        """
        x = self.network(state.unsqueeze(0))
        return torch.mul(x.double(), x.double())

    def int2torch(self,state):
        '''
        convert integer representation into torch representation
        @param state : integer in occupation representation
        @return torch tensor
        '''
        state=torch.from_numpy(self.pe.int2representation(state))
        return state

    def learn(self):
        """" Update the parameters of the network """
        samples = self.sample(self.nb_samples)
        loss = self.compute_loss(samples)
        self.list_loss.append(loss.item())
        # Minimize the loss
        self.network_optimizer.zero_grad()
        loss.backward()
        self.network_optimizer.step()

        self.soft_update(self.network, self.target_network, self.tau)

    def sample(self,n_samples,n_warmup):
        '''
        Samples states using Metropolis-Hasting algorithm
        :param n_samples: number of sampels to collect
        :param n_warmup: number of warm-up states before starting collecting
        :return: list of energies and log probabilities
        '''


        current_state_int = int(self.pe.J[random.randint(0, self.pe.nb_states - 1)]) #select first state randomly
        current_p = self.prob(torch.from_numpy(self.pe.int2representation(current_state_int)).detach().double())

        #Warm-up
        for j in range(0, n_warmup):
            [current_state_int,current_p]=self.get_next_mc_state(current_state_int,current_p)
        # Sample
        log_p = torch.log(self.prob(torch.from_numpy(self.pe.int2representation(current_state_int)))).unsqueeze(0) # Initialize empty list of states
        energies =self.E(current_state_int).unsqueeze(0)
        for j in range(0,n_samples):
            [current_state_int,current_p]=self.get_next_mc_state(current_state_int,current_p)
            log_p=torch.cat((log_p,torch.log(self.prob(torch.from_numpy(self.pe.int2representation(current_state_int)))).unsqueeze(0)))
            energies=torch.cat((energies,self.E(current_state_int).unsqueeze(0)))
            for i in range(0,10):
                [current_state_int, current_p] = self.get_next_mc_state(current_state_int, current_p)

        return energies.squeeze(),log_p.squeeze()

    def get_next_mc_state(self,i,p_i):
        '''
        :param i: binary representation of state i
        :param p_i: probability of state i
        :return: j,p_j selected according to metropolis-hasting method
        '''
        #get new state
        new_state_int = self.pe.J[np.random.randint(0,self.pe.nb_states)]
        with torch.no_grad():
            proposed_p = self.prob(torch.from_numpy(self.pe.int2representation(new_state_int)).detach().double())
        if (proposed_p / p_i > np.random.uniform(0, 1)):
            return new_state_int,proposed_p
        else:
            return i,p_i

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def compute_loss(self,samples):
        loss=0
        #self.sum_i=0
        for s in samples:
            loss+=self.E(s)
        loss/=samples
        return loss

    def E(self,state):
        '''
        Method E computes Re(<c|H|psi>/<c|psi>)
        :param state: (integer) binary occupation state
        :return: "energy" of state.
        '''
        state_H = self.pe.apply_H(int(state)) #compute <c|H| = sum_i state_Hi(0)|state_Hi(1)>
        state_psi = self.network(self.int2torch(state).unsqueeze(0)) #compute <state|psi>
        sum=0

        for c_j,s_j in state_H:
            j_psi = self.network(self.int2torch(s_j).unsqueeze(0))  # compute <j|psi>
            sum+=j_psi*c_j

        return sum/state_psi

    def groud_state(self):
        '''
        :return: ground state of the system
        '''
        gs=[]
        with torch.no_grad():
            for i in np.arange(0,self.pe.nb_states):
                s=self.pe.J[i]
                gs.append(self.network(self.int2torch(s).to("cpu")).numpy())
        return gs

    def quantum_reinforce(self,n_steps,print_every=1):
        '''
        Implement reinforce algorithm-like training scheme
        :param n_steps: number of network updates
        :param print_every: number of steps before writing the loss
        :return: -
        '''

        scores_deque = deque(maxlen=print_every)
        scores = []
        min_data = 64

        for i_episode in range(1, n_steps + 1):
            # Collect nb_samples energies and corresponding log probabilities
            e_locs,log_p = self.sample( n_samples=self.nb_samples,n_warmup=50) #warm-up monte carlo and return the last state
            e_locs_no_grad = e_locs.detach()
            # Compute the loss
            loss = e_locs.mean() + (log_p*e_locs_no_grad).mean() - (log_p.mean())*((e_locs_no_grad).mean())


            self.network_optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm(self.network.parameters(),0.5)
            self.network_optimizer.step()
            energy=e_locs.mean()
            scores_deque.append(-min(energy.detach(),-0.00000001))
            scores.append(energy.detach().item())

            if i_episode % print_every == 0:
                print("*******")
                print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, -np.mean(scores_deque)))
                slope, _, _, p, _ = stats.linregress(range(len(scores_deque)),scores_deque)

                p/=2 #one sided test
                if(p>0.1): # we cannot reject the hypothesis that we didn't improve
                    self.nb_samples+=128 # increase the number of samples
                elif(slope<0): # something is wrong (deque -> data in reverse order)
                    min_data=self.nb_samples+64
                    self.nb_samples+=512
                elif(slope>0.01):
                    self.nb_samples-=64 # Reduce the training set size to try going faster
                    self.nb_samples = max(min_data,self.nb_samples)
                elif(slope<0.001):
                    self.nb_samples+=64

                print('Linear slope {}\t p-value {}'.format(slope,p))
                print('Nb samples = {}'.format(self.nb_samples))
                print("************")
                ## Save the loss and the weights
                torch.save(self.network.state_dict(),"./network_hex_sym_u_"+str(self.u)+"nb_param_"+str(self.network.count_parameters())+"_lr_"+str(self.lr)+".pth")
                pd.DataFrame(scores).to_csv("./loss_hex_sym_llr_u_"+str(self.u)+"nb_param_"+str(self.network.count_parameters())+"_lr_"+str(self.lr)+".csv")
            else:
                print('Episode {}\t Score: {:.2f}'.format(i_episode, energy.detach()))

        return scores

    def get_transition_prob(self,state,new_state):
        '''
        :param state: starting state
        :param new_state: next state
        :return: Pstate->new_state
        '''
        torch_state=torch.from_numpy(self.pe.int2representation(state))
        torch_new_state=torch.from_numpy(self.pe.int2representation(state))
        n_adjacent = np.size(self.pe.get_adjacent_states_from_state(state))

        # We start with the simplest case, where state and new state are different
        if(state!=new_state):
            #we had a probability 1/n_adjacent to select new_state
            # acceptance probability is min(1,p_new/p_old)
            acceptance_probability = torch.min(torch.tensor(1,dtype=torch.double),self.prob(torch_new_state)/self.prob(torch_state))
            return acceptance_probability/n_adjacent
        else:
            # If we stayed on the same state, we compute 1-all possibility
            proba=1
            for st in self.pe.get_adjacent_states_from_state(state):
                # acceptance probability is min(1,p_new/p_old)
                torch_st = torch.from_numpy(self.pe.int2representation(st))

                acceptance_probability = torch.min(torch.tensor(1,dtype=torch.double), self.prob(torch_st) / self.prob(torch_state))
                proba-=acceptance_probability/n_adjacent
            return proba

