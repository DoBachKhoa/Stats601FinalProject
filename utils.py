import math
import random
import numpy as np
from scipy.special import digamma, polygamma


# Discrete distribution from a normal density
def discreteNormal(P, mean, std):
    '''
    Function that truncates and discretizes a normal pdf in to a P-way categorical distribution
    '''
    potentials = np.array([-(mean-i)**2/(std**2) for i in range(P)], dtype='float')
    output = np.exp(potentials)
    output /= np.sum(output)
    return output


# Discrete distribution from a Poisson density
def discretePoisson(P, lam, reverse = False):
    '''
    Function that truncates a Poisson distribution in to a P-way categorical distribution
    '''
    output = [(lam**i)/math.factorial(i) for i in range(P)]
    if reverse: output.reverse()
    output = np.array(output)
    output /= np.sum(output)
    return output