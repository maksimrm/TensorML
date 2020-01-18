import numpy as np



class sampler:
    
    def __init__(self, func, dim = 1, initial = None, bounds = None, shift = 0.01):
        self.func = func
        self.dim = dim
        self.initial = initial
        self.bounds = bounds
        self.shift = shift
        
        
    def gen(self, size, method = "mtr-gas", d = 1):
        
        if method == "mtr-gas":
            return self.mtr_gas(size, d)
    
    
    
    
    def mtr_gas(self, size, d = 1):
        result = np.zeros((size, self.dim))
        if self.initial:
            result[0] = self.initial
        else:
            result[0] = np.random.randn(self.dim)
        i = 1
        prev = result[0]
        while i < size:
            if type(self.bounds) != type(None) and np.random.rand() < self.shift:
                nex= self.bounds[0] + (self.bounds[1] - self.bounds[0])*np.random.rand()
            else :
                nex = prev + np.random.randn(self.dim)/d
            a = self.func(nex)/self.func(prev)
            if a > np.random.random():
                result[i] = nex
                prev = nex
                i+=1
            
        if (self.dim == 1):
            result = result.reshape(size)
        return result