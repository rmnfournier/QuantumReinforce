import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from sympy.combinatorics import Permutation

class Reinforce_Hex_Net(nn.Module):
    def __init__(self,input_size,output_size,architecture,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        """  Builds ACANN network with arbitrary number of hidden layers.

        Arguments
        ----------
        input_size : integer, size of the input
        output_size : integer, size of the output layer
        hidden_layers: list of integers, the sizes of the hidden layers
        """
        super().__init__()
        self.device=device

        # Symmetry of the system
        self.R = torch.as_tensor([[0,0,1,0,0,0,0,0,0,0,0,0],
                                  [1,0,0,0,0,0,0,0,0,0,0,0],
                                  [0,0,0,0,1,0,0,0,0,0,0,0],
                                  [0,1,0,0,0,0,0,0,0,0,0,0],
                                  [0,0,0,0,0,1,0,0,0,0,0,0],
                                  [0,0,0,1,0,0,0,0,0,0,0,0],
                                  [0,0,0,0,0,0,0,0,1,0,0,0],
                                  [0,0,0,0,0,0,1,0,0,0,0,0],
                                  [0,0,0,0,0,0,0,0,0,0,1,0],
                                  [0,0,0,0,0,0,0,1,0,0,0,0],
                                  [0,0,0,0,0,0,0,0,0,0,0,1],
                                  [0,0,0,0,0,0,0,0,0,1,0,0]],device=self.device).double()
        self.M = torch.as_tensor([[0,1,0,0,0,0,0,0,0,0,0,0],
                                  [1,0,0,0,0,0,0,0,0,0,0,0],
                                  [0,0,0,1,0,0,0,0,0,0,0,0],
                                  [0,0,1,0,0,0,0,0,0,0,0,0],
                                  [0,0,0,0,0,1,0,0,0,0,0,0],
                                  [0,0,0,0,1,0,0,0,0,0,0,0],
                                  [0,0,0,0,0,0,0,1,0,0,0,0],
                                  [0,0,0,0,0,0,1,0,0,0,0,0],
                                  [0,0,0,0,0,0,0,0,0,1,0,0],
                                  [0,0,0,0,0,0,0,0,1,0,0,0],
                                  [0,0,0,0,0,0,0,0,0,0,0,1],
                                  [0,0,0,0,0,0,0,0,0,0,1,0]],device=self.device).double()
        self.I = torch.as_tensor([[0,0,0,0,0,0,1,0,0,0,0,0],
                                  [0,0,0,0,0,0,0,1,0,0,0,0],
                                  [0,0,0,0,0,0,0,0,1,0,0,0],
                                  [0,0,0,0,0,0,0,0,0,1,0,0],
                                  [0,0,0,0,0,0,0,0,0,0,1,0],
                                  [0,0,0,0,0,0,0,0,0,0,0,1],
                                  [1,0,0,0,0,0,0,0,0,0,0,0],
                                  [0,1,0,0,0,0,0,0,0,0,0,0],
                                  [0,0,1,0,0,0,0,0,0,0,0,0],
                                  [0,0,0,1,0,0,0,0,0,0,0,0],
                                  [0,0,0,0,1,0,0,0,0,0,0,0],
                                  [0,0,0,0,0,1,0,0,0,0,0,0]],device=self.device).double()

        self.O = torch.as_tensor([[0,0,0,0,1,0,0,0,0,0,0,0],
                                  [0,0,0,0,0,1,0,0,0,0,0,0],
                                  [0,0,1,0,0,0,0,0,0,0,0,0],
                                  [0,0,0,1,0,0,0,0,0,0,0,0],
                                  [1,0,0,0,0,0,0,0,0,0,0,0],
                                  [0,1,0,0,0,0,0,0,0,0,0,0],
                                  [0,0,0,0,0,0,0,0,0,0,1,0],
                                  [0,0,0,0,0,0,0,0,0,0,0,1],
                                  [0,0,0,0,0,0,0,0,1,0,0,0],
                                  [0,0,0,0,0,0,0,0,0,1,0,0],
                                  [0,0,0,0,0,0,1,0,0,0,0,0],
                                  [0,0,0,0,0,0,0,1,0,0,0,0]],device=self.device).double()
        self.Ts = []
        for r in np.arange(6):
            # Compute the current rotation matrix
            R = self.R
            for _ in np.arange(r):
                R=R.mm(self.R)

            for m in np.arange(2):
                # Compute the current mirror
                M = self.M
                for _ in np.arange(m):
                    M=M.mm(self.M)
                for i in np.arange(2):
                    # Compute the current spin inversion matrix
                    I = self.I
                    for _ in np.arange(i):
                        I = I.mm(self.I)

                    for o in np.arange(2):
                        O = self.O
                        for _ in np.arange(o):
                            O = O.mm(self.O)
                        self.Ts.append(R.mm(M.mm(I.mm(O))))

            
        self.group_elements = len(self.Ts)
        # Add the first layer : input_size into the first hidden layer
        self.conv1 = nn.ModuleList([nn.Linear(input_size,architecture[0]).to(device)])
        # Add the other layers
        layers_sizes = zip(architecture[:-1],architecture[1:])
        self.conv1.extend([nn.Linear(h1,h2).to(device) for h1,h2 in layers_sizes])
        #self.conv1_output=nn.Linear (conv_1[-1],dense_end[0])
        self.conv1_output = nn.Linear(architecture[-1], output_size)
        # Add the first layer : input_size into the first hidden layer
        #self.dense = nn.ModuleList()
        # Add the other layers
        #dense_end[0]*=self.group_elements # because of the rolling
        #layers_sizes = zip(dense_end[:-1],dense_end[1:])
        #self.dense.extend([nn.Linear(h1,h2).to(device) for h1,h2 in layers_sizes])
        #self.dense_output=nn.Linear (dense_end[-1],output_size)


    def forward(self,x):
        '''
        Implement the symmetries of the state
        :param x: nsamplesx12 vector, where the first (last) 6 components correspond to a single spin
        :return: <x|psi>
        '''
        x.to(self.device)

        # We start by implementing the c3v+I slicing
        # output -> inv nsamples*24x12
        inv = torch.zeros(size=(self.group_elements * x.shape[0], x.shape[1]))
        for i, T in enumerate(self.Ts):
            inv[i::self.group_elements, :] = x.mm(T)
        # We then apply the first convolutional network to each element
        for l in self.conv1:
            inv = torch.relu(l.forward(inv))
        inv=self.conv1_output.forward(inv)
        # Then we apply the rolling step
        #index = np.asarray([np.roll(np.arange(self.group_elements) + i * self.group_elements, j) for i in np.arange(x.shape[0]) for j in np.arange(self.group_elements)])
        #inv2 = inv[index,:].flatten(1)
        # Now we can get perform the dense operations
        #  We then apply the first convolutional network to each element
        #for l in self.dense:
        #    inv2 = torch.relu(l.forward(inv2))
        #inv = self.dense_output.forward(inv2)
        # Finally, we perform the pooling
        pool  = torch.nn.AvgPool1d(self.group_elements)
        value = pool.forward(inv.flatten().unsqueeze(0).unsqueeze(1)).squeeze()

        # We still have to define the sign before the value, because of the anticommutation rules
        state = x.cpu().detach().numpy()[0]
        hex_order = [0,1,3,5,4,2,6,7,9,11,10,8]
        # keep track of the current order
        order = state * np.arange(1, 13)
        # then, put it in the hex order
        state_in_hex = order[hex_order]
        # get the non-0 elements
        state_in_hex = state_in_hex[np.where(state_in_hex > 0.1)]
        # get the ordered indices
        order = np.argsort(state_in_hex)
        # create a permutation
        perm = Permutation(order)
        # print the sign
        sign = perm.signature()

        return sign*value


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)