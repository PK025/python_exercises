#!/usr/bin/python
"""
@author: PK025

Basic implementation of descrete Fourier transform and inverse transform.

"""

import math
import numpy as np


def DFT(X):
    """
    Descrete Fourier transform

    Parameters
    ----------
    X : numpy.array
        Input function ,one dimentional

    Returns
    -------
    Re : numpy.array
        Real part of calculated transform
    Im : numpy.array
        Imaginary part of calculated transform

    """
    N = X.shape[0]
    Wi = np.arange(0, N).reshape((1, N))*np.arange(0, N).reshape((N, 1))
    Wcos = np.cos((2*math.pi/N)*Wi)
    Wsin = np.sin((2*math.pi/N)*Wi)
    Re = np.matmul(Wcos,X.reshape((N, 1)))
    Im = -np.matmul(Wsin,X.reshape((N, 1)))
    return (Re, Im)
    

def IDFT(Re, Im):
    """
    Inverse descrete Fourier transform

    Parameters
    ----------
    Re : numpy.array
        Real part of Fourier transform
    Im : numpy.array
        Imaginary part of Fourier transform

    Returns
    -------
    X : numpy.array
        Result of inverse transform

    """
    N = Re.shape[0]
    Wi = np.arange(0, N).reshape((1, N))*np.arange(0, N).reshape((N, 1))
    Wcos = np.cos((2*math.pi/N)*Wi)
    Wsin = np.sin((2*math.pi/N)*Wi)
    X = np.matmul(Wcos,Re.reshape(N, 1)) - np.matmul(Wsin,Im.reshape(N, 1))
    X = X/N
    return X
    

