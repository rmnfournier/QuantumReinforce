import numpy as np
from scipy import special
import gmpy2
import cmath
import random

class Physics_Engine:
    """ Handles the physics of lattices for Machine Learning Purpose """

    def __init__(self, matrix_length, nb_flavors, filling,k,U):
        if matrix_length==2:
            self.lattice = np.array([[1, 2], [0,0]]) # Currently, we only consider a 2 sites lattice _ _
            self.lattice_neighbors = [[0,1],[0,-1]] # Positions of the neighbors (either left of right) Be careful, first component is y, second x!!
        elif matrix_length==4:
            self.lattice = np.array([[0,1,2,0],[3,0,0,4],[0,5,6,0],[0,0,0,0] ])
            self.lattice_neighbors = [[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]] # Be careful, first component is y, second x!!

        elif matrix_length==6:
            self.lattice = np.array([[0,0,0,1,2,0],[0,3,4,0,0,5],[6,0,0,7,8,0],[0,9,10,0,0,11],[0,0,0,12,13,0]])
            self.lattice_neighbors = [[0, 1], [0, -1], [1, 1], [1, -1], [-1, 1],[-1, -1]]  # Be careful, first component is y, second x!!

        # Lattice vector
        a=1.42*10**(-10)
        self.a1=[a*3,a*cmath.sqrt(3)]
        self.a2=[a*3,-a*cmath.sqrt(3)]
        self.nb_sites = int(np.size(np.nonzero(self.lattice)[0]))
        self.t = 1
        self.U = U
        self.k=k

        # Compute the number of elements for the current filling
        self.nb_electrons = int(nb_flavors * self.nb_sites * filling)
        self.nb_flavors = int(nb_flavors)
        self.filling = int(filling)
        self.nb_states = int(special.binom(self.nb_sites * nb_flavors, self.nb_electrons))

        # Build the LookUp Table
        self.J_up = np.zeros([int(pow(2, self.nb_flavors / 2 * self.nb_sites)), 1])
        self.J_down = np.zeros([int(pow(2, self.nb_flavors / 2 * self.nb_sites)), 1])
        self.J = np.zeros([self.nb_states, 1])
        self.build_lookuptable()
        self.adjacent_states = []
        #self.set_list_adjacent_states()

    ### Lattice
    def find_site(self, i):
        """
        @param i:integer , site to be found
        """
        return np.where(self.lattice == i)

    def get_neighbors(self, i):
        """
        @param i:integer , site of interest
        """
        [x, y] = self.find_site(i)
        neig = []
        for n in self.lattice_neighbors:
            # if we stayed in the grid
            if (x[0] + n[0] >= 0 and x[0] + n[0] < len(self.lattice) and y[0] + n[1] >= 0 and y[0] + n[1] < len(
                    self.lattice[0])):
                # if the site is not empty
                tmp=self.lattice[x[0] + n[0]][y[0] + n[1]]
                tmpx=x[0] + n[0]
                if self.lattice[x[0] + n[0]][y[0] + n[1]] > 0:
                    neig.append(self.lattice[x[0] + n[0]][y[0] + n[1]])
        return neig

    def build_lattice(self, matrix_length):
        # Build a matrix containing 1 where the lattice has a site and 0 elsewhere
        for i in range(0, matrix_length):
            occupied = 1 - i % 2  # if the line starts with a site or note
            for j in range(1, matrix_length + 1):
                # For HC lattice, we have site-site-empty-empty-site-site etc
                if j % 2 == 0:
                    occupied = 1 - occupied
                self.lattice[i][j - 1] = occupied

    ### Algebra
    def build_lookuptable(self):
        """ Build a clever basis """
        j_down = 0
        for n_down in range(0, self.nb_electrons + 1):
            nb_states = special.binom(self.nb_flavors / 2 * self.nb_sites, self.nb_electrons - n_down)
            for down_part in range(0, int(pow(2, self.nb_flavors / 2 * self.nb_sites))):
                j_up = 0
                if gmpy2.popcount(down_part) == n_down:
                    for up_part in range(0, int(pow(2, self.nb_flavors / 2 * self.nb_sites))):
                        if gmpy2.popcount(up_part) == self.nb_electrons - n_down:
                            self.J_up[up_part] = j_up
                            self.J_down[down_part] = j_down
                            self.J[int(j_up + j_down)] = pow(2,
                                                             self.nb_flavors / 2 * self.nb_sites) * down_part + up_part
                            j_up += 1
                    j_up = 0
                    j_down += nb_states

    def pos_to_index(self, i):
        """@param(i): state in occupation representaiton"""
        return int(self.J_up[(int(i % pow(2, self.nb_flavors / 2 * self.nb_sites)))] + self.J_down[
            (int(i / pow(2, self.nb_flavors / 2 * self.nb_sites)))])

    ## Hamiltonian
    def apply_H_pot(self, i):
        """
        @param i : unsigned integer, state in occupation representation.
        """
        potential_energy = 0
        ## loop over the sites
        for site in range(0, self.nb_sites):
            if (gmpy2.bit_test(i, site) and gmpy2.bit_test(i, site + self.nb_sites)):
                potential_energy += self.U
        return [[potential_energy, i]]

    def apply_H_cin(self, i):
        """
        @param i : unsigned integer, state in occupation representation.
        """
        targets = []
        # for each site
        for site in range(1, self.nb_sites + 1):
            # Perform the following tasks for both spins
            for spin in [site, site + self.nb_sites]:
                # if there is a current spin at site
                if (gmpy2.bit_test(i, spin - 1)):
                    # loop over each neighbors
                    for neighbor in self.get_neighbors(site):
                        # add element to the answer if site is vacant
                        if (not gmpy2.bit_test(i, int(spin - site + neighbor - 1))):
                            # count the number of electrons between the creation and annihilation sites
                            create_at = int(spin - site + neighbor - 1)
                            destroy_at = int(spin-1)
                            n_el =0
                            for c in np.arange(min(create_at,destroy_at)+1,max(create_at,destroy_at) ):
                                if gmpy2.bit_test(i,int(c)):
                                    n_el+=1
                            targets.append(
                                [((-1)**(1+n_el))*self.t, gmpy2.bit_flip(gmpy2.bit_flip(i, spin - 1), int(spin - site + neighbor - 1))])
        return targets

    def apply_H(self, i):
        """
        @param i : unsigned integer, state in occupation representation.
        """
        return self.apply_H_cin(i) + self.apply_H_pot(i)


    def int2representation(self,i):
        representation=np.zeros([self.nb_sites*self.nb_flavors])
        for r in np.arange(0,np.size(representation)):
            if(gmpy2.bit_test(int(i),int(r))):
                representation[r]=1
        return representation

    def get_random_adjacent_state(self,i):
        '''
        :param i: (integer) state which is in the binary representation state
        :return: state j which is obtained by permuting one electron and one hole from state i
        '''
        pos_e=random.randint(1,self.nb_electrons) # we will move the pos_e.th. electron
        pos_h = random.randint(1,self.nb_sites*self.nb_flavors-self.nb_electrons) # we will permute the electron with the pos_h th. hole

        count_e=0
        count_h=0 #count the number of visited bit with electrons and holes

        has_flipped = False # if we have already made a move

        for site in range(0,self.nb_flavors*self.nb_sites): #we loop over each bit of i.
            #update the number of visited bits with electrons and holes
            if(gmpy2.bit_test(i,site)):
                count_e+=1
            else:
                count_h+=1
            #Flip the bit if it's the correct position
            if(count_e==pos_e):
                i=gmpy2.bit_flip(i,site)
                count_e+=1 #update count_e to not enter the condition on the next loop
                if not has_flipped: #if it's the first flip
                    has_flipped=True
                else: #if we already made a move, then we can exist the loop
                    break
            elif(count_h==pos_h):
                i=gmpy2.bit_flip(i,site)
                count_h+=1
                if not has_flipped: #if it's the first flip
                    has_flipped=True
                else: #if we already made a move, then we can exist the loop
                    break
        return i

    def set_list_adjacent_states(self):
        '''
            initialize list_adjacent_states
        '''
        self.adjacent_states=[]
        for i in range(0,self.nb_states):
            current_list=[]
            current_state=int(self.J[i]) # get the ith. state
            for j in range(0,self.nb_states):
                test_state=int(self.J[j])
                # check if the site differs at 2 sites
                bit_dif=current_state^test_state #this number contains 1 at each bit where the states differs
                if(gmpy2.popcount(bit_dif)==2): #if we have only 2 different position
                    current_list.append(test_state)

            self.adjacent_states.append(current_list)

    def get_adjacent_states_from_state(self,i):
        '''
        :param i: binary representation of a state
        :return: list of its adjacent states
        '''
        return self.adjacent_states[self.pos_to_index(i)]

    def apply_c(self,a,i,dagger=False):
        '''
        compute C(?dagger)(a)|i>
        :param i: (int) binary representation of state i in occupation state
        :param a: (int) number of the site(+spin) on which we want to apply cdagger
        :param dagger: (bool) if True, return cdagger|i>, othervise c|i>
        :return: (int,int) 1 or -1, binary representation of state cd|i> in occupation state
        '''
        #Get the ath state
        ath_state = gmpy2.bit_test(i,a)
        #if the site is empty and we want to destroy of occupied and we want to create
        if(ath_state==dagger):
            return [0,0]
        # Othervise
        else:
            # count the number of occupied states before a to get the phase
            nb_bits_on_before_a = gmpy2.popcount(i%(1<<a))
            # flip the ath bit
            new_state=gmpy2.bit_flip(i,a)
            # return the phase and the new state
            return [(-1)**nb_bits_on_before_a,new_state]

    def apply_c_dagger(self,a,i):
        '''
        compute C(dagger)(a)|i>
        this method is just another way of calling apply_C(a,i,dagger=True)
        :param i: (int) binary representation of state i in occupation state
        :param a: (int) number of the site(+spin) on which we want to apply cdagger
        '''
        return self.apply_c(a,i,dagger=True)

    def get_H(self):
        '''
        :return: Hamiltonian of the system
        '''
        H=np.zeros([self.nb_states,self.nb_states])
        for i in np.arange(0,self.nb_states):
            tmps = self.apply_H(int(self.J[i]))
            for tmp in tmps:
                j=self.pos_to_index(int(tmp[1]))
                H[i,j]=tmp[0]
        return H