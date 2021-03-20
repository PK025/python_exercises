#!/usr/bin/python


import math
import numpy as np



def DFT(X):
    N = X.shape[0]
    Wi = np.arange(0, N).reshape((1, N))*np.arange(0, N).reshape((N, 1))
    Wcos = np.cos((2*math.pi/N)*Wi)
    Wsin = np.sin((2*math.pi/N)*Wi)
    Re = np.matmul(Wcos,X.reshape((N, 1)))
    Im = -np.matmul(Wsin,X.reshape((N, 1)))
    return (Re, Im)
    

def IDFT(Re, Im):
    N = Re.shape[0]
    Wi = np.arange(0, N).reshape((1, N))*np.arange(0, N).reshape((N, 1))
    Wcos = np.cos((2*math.pi/N)*Wi)
    Wsin = np.sin((2*math.pi/N)*Wi)
    X = np.matmul(Wcos,Re.reshape(N, 1)) - np.matmul(Wsin,Im.reshape(N, 1))
    X = X/N
    return X
    

