import math
import random
import numpy as np
from scipy.special import digamma, polygamma


# Discrete distribution from a normal density
def discreteNormal(P, mean=None, std=None):
    '''
    Define a P-way categorical distribution in 1D
        by truncating and discretizing a the density of a 1D normaly distribution
        with the given mean and standard deviation.
    
    Parameters
    ----------
    P : int, positive
        number of categories
    mean : float (default = None)
        mean of the gaussian distribution
    std : float, positive (default = None)
        standard deviation of the gaussian distribution

    Default values are chosen so that the default distribution scales well for large P.
    If not given, mean is taken to be P/2.
    If not given, std is taken to be P.

    Return
    ------
    array-like of shape (P,)
        probability function of the categorical distribution
        represented by an array
    '''
    # Type and positivity check for P
    if type(P) != int: raise ValueError('P has to be an integer')
    if P <= 0: raise ValueError('P has to be positive')
    
    # Type and positivity check for mean and std
    if mean == None: mean = P/2.
    if std  == None: std  = P
    if std  <= 0: raise ValueError('Standard deviation has to be positive')

    # Calculate the P-way categorical distribution probabilities
    potentials = np.array([-(mean-i)**2/(std**2) for i in range(P)], dtype='float')
    output = np.exp(potentials)
    output /= np.sum(output)
    return output


# Discrete distribution from a Poisson density
def discretePoisson(P, lam=1., n_0=0, reverse=False):
    '''
    Define a P-way categorical distribution in 1D
        by truncating the Poisson distribution
        with the given lambda (lam) and n_0 indicating the window being taken
        with option to reversed the returning output if needed.
    The i-th probababily of the output (if not reversed) or the P-1-i th probability
        (if reversed) will scale with lam**(n_0+i)/(n_0+i)
        for all i from 0 to P-1.
    This means that n_0 has to be nonnegative.
    
    Parameters
    ----------
    P : int, positive
        number of categories
    lam : float (default = 1)
        lambda value for the Poisson distribution
    n_0 : int, nonnegative (default = 0)
        off set for the taking window of the Poisson distribution
    reverse : bool (default = False)
        reverse option for output.

    Return
    ------
    array-like of shape (P,)
        probability function of the categorical distribution
        represented by an array
    '''
    # Type check for P , lam and n_0
    if type(P) != int: raise ValueError('P has to be an integer')
    if P <= 0: raise ValueError('P has to be a positive integer')
    if type(lam) not in [float, int]: raise ValueError('`lam` variable has to be a float')
    if lam <= 1e-9: raise ValueError('`lam` has to be positive')
    if type(n_0) != int: raise ValueError('n_0 has to be an integer')
    if n_0 < 0: raise ValueError('Offset value n_0 has to be nonnegative')
    if type(reverse) != bool and (reverse not in [0, 1]):
        raise ValueError('`reverse` is not a boolean')

    # Calculate the P-way categorical distribution probabilities
    output = [(lam**(n_0+i))/math.factorial(n_0+i) for i in range(P)]
    if reverse: output.reverse()
    output = np.array(output)
    output /= np.sum(output)
    return output

# Trivial discrete pobability
def discreteInput(l):
    return np.array(l)/np.sum(l)
