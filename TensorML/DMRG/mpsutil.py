import numpy as np
from numpy import linalg as LA
from abc import ABC, abstractmethod
from ncon import ncon


class AbstractInitilizer(ABC):
    
    SIGMA_0 = np.eye(2)
    SIGMA_X = np.array([[0,1],[1,0]])
    SIGMA_Y = np.array([[0, - 1j],[1j, 0]])
    SIGMA_Z = np.array([[1, 0],[0, -1]])
    ZERO = np.zeros((2,2))
    SIGMA_PLUS = np.sqrt(2)*np.array([[0, 0],[1, 0]])
    SIGMA_MINUS = np.sqrt(2)*np.array([[0, 1],[0, 0]])
    
    def __init__(self, num_of_particles, bond_dim, phys_dim):
        self.num_of_particles = num_of_particles
        self.bond_dim = bond_dim
        self.phys_dim = phys_dim
    
    
    @abstractmethod
    def norm(self):
        """DEFENITION OF TENSOR NORMS"""
        pass
    
    @abstractmethod
    def __repr__(self):
        """SUMMARY OF INITILIALIZED DATA STRUCTURES"""
        pass
    
    @abstractmethod
    def __call__(self):
        """ABILITY TO CALL AN OBJECT"""
        pass
    
    
class StateInitializer(AbstractInitilizer):
    """OBC - OPEN BOUNDARY CONDITIONS"""
    def __call__(self, normalize=False):
        left_node_mps = np.random.randn(self.phys_dim, self.bond_dim)\
                        + 1j * np.random.randn(self.phys_dim, self.bond_dim)
        
        right_node_mps = np.random.randn(self.bond_dim, self.phys_dim)\
                        + 1j * np.random.randn(self.bond_dim, self.phys_dim)
    
        bulk_node_mps = [np.random.randn(self.bond_dim, self.phys_dim, self.bond_dim)\
                        + 1j * np.random.randn(self.bond_dim, self.phys_dim, self.bond_dim)
                         for _ in range(int(self.num_of_particles) - 2)]
        
        short_list = [left_node_mps] + bulk_node_mps + [right_node_mps]
        
        if normalize:
            norm_step = ncon([short_list[0],short_list[0].conj()], [[1,-1], [1,-2]])
            
            for i in range(1, self.num_of_particles-1):
                norm_step = ncon([norm_step, short_list[i], short_list[i].conj()],
                                [[1,2], [1,3,-1], [2,3,-2]])
            
            norm = np.abs(ncon([norm_step, short_list[-1], short_list[-1].conj()], [[1,2], [1,3], [2,3]]))
            short_list[0] = short_list[0]/np.sqrt(norm)
        
        return short_list
    
    def __repr__(self):
        """SUMMARY OF INITILIALIZED DATA STRUCTURES"""
        pass
    
    def norm(self):
        """DEFENITION OF TENSOR NORMS"""
        pass
    
    
class OperatorInitializer(AbstractInitilizer):
    """SIMPLE 1D ISING"""
    def __call__(self, interact_J, interact_field_h=0):
        left_node_mpo = np.zeros((self.phys_dim, self.phys_dim, self.bond_dim))
        left_node_mpo[:,:,0] = self.ZERO
        left_node_mpo[:,:,1] = interact_J * self.SIGMA_Z
        left_node_mpo[:,:,2] = self.SIGMA_0
        
        right_node_mpo = np.zeros((self.bond_dim, self.phys_dim, self.phys_dim))
        right_node_mpo[0] = self.SIGMA_0
        right_node_mpo[1] = self.SIGMA_Z
        right_node_mpo[2] = self.ZERO
        
        bulk_node_mpo = np.zeros((self.bond_dim, self.phys_dim, self.phys_dim, self.bond_dim))
        bulk_node_mpo[0,:,:,0] = self.SIGMA_0
        bulk_node_mpo[1,:,:,0] = self.SIGMA_Z
        bulk_node_mpo[2,:,:,1] = interact_J * self.SIGMA_Z 
        bulk_node_mpo[2,:,:,2] = self.SIGMA_0
        
        return left_node_mpo, bulk_node_mpo, right_node_mpo
    
    def __repr__(self):
        """SUMMARY OF INITILIALIZED DATA STRUCTURES"""
        pass
    
    def norm(self):
        """DEFENITION OF TENSOR NORMS"""
        pass
    

class XXInitializer(AbstractInitilizer):
    
    def __call__(self):
        left_mpo_bond = np.array([1,0,0,0]).reshape(4,1,1)
        right_mpo_bond = np.array([0,0,0,1]).reshape(4,1,1)
        
        bulk_node_mpo = np.zeros((self.bond_dim, self.bond_dim, self.phys_dim, self.phys_dim));
        bulk_node_mpo[0,0] = self.SIGMA_0
        bulk_node_mpo[0,1] = self.SIGMA_MINUS
        bulk_node_mpo[0,2] = self.SIGMA_PLUS
        bulk_node_mpo[1,3] = self.SIGMA_PLUS
        bulk_node_mpo[2,3] = self.SIGMA_MINUS
        bulk_node_mpo[3,3] = self.SIGMA_0
        
        return left_mpo_bond, bulk_node_mpo, right_mpo_bond
    
    def __repr__(self):
        return "[ sigma0  sigma-  sigma+   0    ]\n" +\
                "[   0       0       0    sigma+ ]\n" +\
                "[   0       0       0    sigma- ]\n" +\
                "[   0       0       0    sigma0 ]"
    
    def norm(self):
        pass
    
    
class VariableBondStateInit(AbstractInitilizer):
    
    def __call__(self, chi):
        self.bond_dim = min(chi, self.bond_dim)
        state = [np.random.rand(1, self.phys_dim, self.bond_dim)]
        for k in range(1, self.num_of_particles):
            state.append(
                np.random.rand(self.bond_dim, self.phys_dim,
                    min(chi, self.bond_dim * self.phys_dim,
                        self.phys_dim**(self.num_of_particles-k-1)))
            )
            
            self.bond_dim = state[k].shape[2]
        
        return state
            
    def __repr__(self):
        pass
    
    def norm(self):
        pass

    
'''
class CanonicalMPS:
    
    def __init__(self, short_list):
        self.short_list = short_list
        mps_length = len(short_list)
        
        right_to_left = [ncon([short_list[-1], short_list[-1].conj()], [[-1,1], [-2,1]])]
        
        for i in range(1, mps_length-1):
            right_to_left.append(
                    ncon([short_list[-i-1], short_list[-i-1].conj(), right_to_left[i-1]],
                            [[-1,1,3],[-2,2,3],[1,2]])
                )
        
        #LEFT TO RIGHT CONV. + FINAL CONV.#
        link_matrices = [None] * (mps_length - 1)
        node_list = [None] * len(short_list)
        state_list = [None] * (2 * mps_length - 1)
        for i in range(mps_length - 1):  
            if i==0:
                conv_train = ncon([short_list[0], short_list[0].conj()], [[-1,1],[-2,1]])
            else:
                conv_train = ncon([link_matrices[i-1], link_matrices[i-1].conj(),
                    ncon([node_list[i], node_list[i].conj()], [[-1,-3,1],[-2,-4,1]])], [[1,2],[1,3],[2,3,-1,-2]])
                    
            """LEFT DECOMPOSITION FOR STEP i"""
            lmbd_left, U_left = LA.eigh(conv_train)
            print(lmbd_left)
            lmbd_left = np.sqrt(np.abs((lmbd_left)))
            X_left = U_left @ np.diag(lmbd_left) @ U_left.conj().T
            X_left_inv = U_left @ np.diag((1/(lmbd_left))) @  U_left.conj().T
            
            """RIGHT DECOMPOSITION (USING RIGHT TO LEFT) FOR STEP i"""
            lmbd_right, U_right = LA.eigh(right_to_left[-1-i])
            lmbd_right = np.sqrt(np.abs((lmbd_right)))
            X_right = U_right @ np.diag(lmbd_right) @ U_right.conj().T
            X_right_inv =  U_right @ np.diag((1/(lmbd_right))) @ U_right.conj().T
            
            """LINK MATRICES LIST"""
            link_matrix = X_right @ X_left 
            #U,link_matrix,V = LA.svd(link_matrix)
            link_matrices[i] = link_matrix
            if  i==0:
                print("X train vs eye: ", LA.norm(X_left_inv@X_left@X_left@X_left_inv - np.eye(len(X_left))))
                print("X**2 vs |short_list|**2: ", LA.norm(X_left@X_left - short_list[0]@short_list[0].conj().T))
                print("XslslX vs eye: ", LA.norm(X_left_inv@short_list[0]@short_list[0].conj().T@X_left_inv - np.eye(len(X_left))))
                print("X train norm", LA.norm(X_left_inv@X_left@X_left@X_left_inv))
                print("XslslX norm: ", LA.norm(X_left_inv@short_list[0]@short_list[0].conj().T@X_left_inv))
                print("=======================================")
                print(X_left@X_left)
                print("=======================================")
                print(short_list[0]@short_list[0].conj().T)
                node_list[i] = ncon([short_list[i],X_left_inv], [[1,-2],[-1,1]])
                node_list[i+1] = ncon([X_right_inv, short_list[i+1]], [[-1,1],[1,-2,-3]])
            elif i == mps_length - 2:
                node_list[i] = ncon([node_list[i],X_left_inv], [[-1,1,-3],[-2,1]])
                node_list[i+1] = ncon([X_right_inv,short_list[i+1]], [[-1,1],[1,-2]])
            else:
                node_list[i] = ncon([node_list[i],X_left_inv], [[-1,1,-3],[-2,1]])
                node_list[i+1] = ncon([X_right_inv,short_list[i+1]], [[-1,1],[1,-2,-3]])
                """STATE LIST NODE TENSORS + LINK MATRICES"""
            
        """ASSEMBLING THE CANONICAL STATE"""
        for j in range(len(state_list)):
            if j%2 == 0:
                state_list[j] = node_list[int(j/2)]
            else:
                state_list[j] = link_matrices[int((j-1)/2)]
                    
        """ ATRIBUTE - STATE"""  
        self.state = state_list
        self.links = link_matrices
        self.nodes = node_list

       
    
    def __call__(self):
        """CALLABLE"""
        return self.state
'''
