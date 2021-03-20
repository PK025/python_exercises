#!/usr/bin/python

import numpy as np

class BackpropagationNetwork:
    
    
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        self.w1 = np.ones((input_size,input_size), dtype=float)
        self.w1 -= 2*np.random.rand(input_size,input_size)
        self.b1 = np.zeros(input_size, dtype=float)
        self.o1 = np.zeros(input_size, dtype=float)
        
        self.w2 = np.ones((output_size,input_size), dtype=float)
        self.w2 -= 2*np.random.rand(output_size,input_size)
        self.b2 = np.zeros(output_size, dtype=float)
        self.o2 = np.zeros(output_size, dtype=float)
        
        self.mi = 1.0
        

    def train(self, x, t):
        if(x.shape[0] != t.shape[0]):
            raise Exception('Arrays have to be equal length')
        
        if(x.shape[1]!=self.input_size):
            raise Exception('Input array size does not fit')
           
        if(t.shape[1]!=self.output_size):
            raise Exception('Result array does not fit')
        
        E = 0
        
        for i in range(x.shape[0]):
            net1 = x[i]@self.w1.T + self.b1
            o1 = 1/(1+np.exp(-net1)).reshape(1,self.input_size)
            net2 = o1@self.w2.T + self.b2
            o2 = 1/(1+np.exp(-net2)).reshape(1,self.output_size)
            
            s2 = np.multiply((o2-t[i]), np.multiply(o2, 1-o2))
            s1 = np.multiply(np.matmul(s2,self.w2), np.multiply(o1, 1-o1))
            s1 = s1.reshape(1,self.input_size)
            s2 = s2.reshape(1,self.output_size)
            
            self.w2 = self.w2 - self.mi*(s2.T@o1)
            self.b2 = self.b2 - self.mi*s2
            
            self.w1 = self.w1 - self.mi*(s1.T@x[i].reshape(1,self.input_size))
            self.b1 = self.b1 - self.mi*s1
            
            E += np.multiply(t[i]-o2, t[i]-o2).sum()
        
        return E
            
    
    def predict(self, x):
    
        if(x.shape[1]!=self.input_size):
            raise Exception('Input array size does not fit')
            
        y = np.zeros((x.shape[0], self.output_size))
        for i in range(x.shape[0]):
            
            net1 = x[i]@self.w1.T + self.b1
            o1 = 1/(1+np.exp(-net1))
            
            net2 = o1@self.w2.T + self.b2
            o2 = 1/(1+np.exp(-net2))
            
            y[i] = o2
            
        return y
    
    
    def set_mi(self, mi):
        self.mi = mi
            


