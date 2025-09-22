import tqdm
import math
import numpy as np
from scipy.special import digamma, polygamma


# Algorithm 1: Newton Raphson update
def NewtonRaphson(alpha, g, h, z, safecheck=False):
    '''
    Newton Rahpson update in linear time for Hessian matrix with special structure.
    Algorithm 1 in the report.
    When the hessian matrix H has special structure
        H = diag(h) + 1 @ z @ 1.T
        then the NR update alpha <- alpha - H^(-1) @ g
        can be computed in linear time.

    Parameters
    ----------
    alpha : numpy array of length K
        K being the number of topics in the main algorithm
        array to update
    g : numpy array of length K
        computed gradient of alpha
    h : numpy array of length K
    z : float 
        h and z compose the hessian matrix H of alpha
        H = diag(h) + 1 @ z @ 1.T
    safecheck : bool (default False)
        correctness checking option

    Return
    ------
    array-like of length K
        updated value of alpha
    '''
    # Type check
    if safecheck:
        narray = type(np.array([0.]))
        if not isinstance(alpha, narray): raise ValueError('alpha has to have type numpy.array')
        if not isinstance(g, narray): raise ValueError('g has to have type numpy.array')
        if not isinstance(h, narray): raise ValueError('h has to have type numpy.array')
        if not isinstance(z, float): raise ValueError('z has to be a float')
        if not alpha.ndim == g.ndim == h.ndim == 1:
            raise ValueError('alpha, g, h have to be 1D')
        if not alpha.shape[0] == g.shape[0] == h.shape[0]:
            raise ValueError('alpha, g, h have to have simmilar length')

    # Calculates c in linear time
    c = np.sum(g/h) / (1/z + np.sum(1./h)) if (np.abs(z) > 1e-9) else 0.

    # Returns new alpha in linear time
    return alpha - ((g-c)/h)


# Algorithm 2: Variational inference
def VariationalInference(ww, alpha, beta, num_iter=10, safecheck=False):
    '''
    Variational Inference algorithm that approximates the posteriors
        of a LDA model with parameter `alpha` and `beta`
        given the words `ww`.
    Algorithm 2 in the report.
    Deterministically initiallize the variational parameters `gamma` and `phi`
        and run coordinate ascent to optimize these variational parameters

    Parameters
    ----------
    ww : array-like, shape (N,)
        list of words with length N
    alpha : array-like, shape (K,)
        K being the number of topics in the model
        vector of topic mixing prior
    beta : array-like, shape (K, P)
        P being the number of words in the dictionary
        matrix of word distribution per topic
    num_iter: int
        number of coordinate ascent loop

    Return
    ------
    gamma : array-like of shape (K,)
        variational parameter
    phi : array-like of shape (N, K)
        variational parameter
    '''
    # Dimension and shape check of ww, alpha, and beta:
    if safecheck:
        if ww.ndim != 1: raise ValueError('ww has to be a 1D vector')
        if alpha.ndim != 1: raise ValueError('alpha has to be a 1D vector')
        if beta.ndim != 2: raise ValueError('beta has to be a 1D vector')
        if alpha.shape[0] != beta.shape[0]: raise ValueError('alpha and beta length unmatched')
        if type(num_iter) != int or num_iter < 1:
            raise ValueError('Invalid number of iteration num_iter')

    # Initialization
    N = ww.shape[0]
    K, P = beta.shape
    gamma = np.full((K,),1/K)
    phi = np.full((N, K),1/K)

    # Coordinate ascent loop
    for _ in range(num_iter):
        for n in range(N):
            for i in range(K):
                phi[n][i] = beta[i][ww[n]]*np.exp(digamma(gamma[i]))
            phi[n] /= np.sum(phi[n])
        gamma = alpha + np.sum(phi, axis = 0, keepdims = False)

    return gamma, phi    


# Algorithm 3: Parameter estimation
def ParameterEstimation(wws, K, P, num_iter_VI=10, num_iter_NR=3, num_iter_EM=10, printing = True):
    '''
    Variational MLE parameter estimation algorithm.
    Implement Variational EM, together with Algorithm 1 and 2 in the report.
    Algorithm 3 in the report.

    Parameters
    ----------
    wws : array-like of array of integers
        list of vectors representing words
    K : int, positive
        number of topics
    P : int, positive
        number of words
    num_iter_VI : int (default 10)
        number of VI iteration for algorithm 2
    num_iter_NR : int (default 3)
        number of Newton Raphson interation for algorithm 1
    num_iter_EM : int (default 10)
        number of EM iteration of the main loop
    printing : bool (default True)
        option to print out the loading bar using tqdm
    
    Return
    ------
    array-like of shape (K,)
        optimized estimation of alpha
    array-like of shape (K, P)
        optimized estimation of beta
    '''
    # Initialization
    M = len(wws)
    alpha = np.random.exponential(scale=2., size=K)
    beta = np.random.dirichlet([0.5]*P, size=K)

    # Variational EM main loop
    for _ in (tqdm.tqdm(range(num_iter_EM)) if printing else range(num_iter_EM)):

        # E step
        gammas = []
        phis = []
        for ww in wws:
            gamma, phi = VariationalInference(ww, alpha, beta, num_iter=num_iter_VI)
            gammas.append(gamma)
            phis.append(phi)

        # M step for alpha
        for _ in range(num_iter_NR):
            g = M*(digamma(np.sum(alpha))-digamma(alpha))
            for m in range(M):
                g -= (digamma(np.sum(gammas[m]))-digamma(gammas[m]))
            h = - M * polygamma(1, alpha)
            z = M * polygamma(1, np.sum(alpha))
            alpha = np.maximum(1e-9, NewtonRaphson(alpha, g, h, z))

        # M step for beta
        beta = np.zeros_like(beta, dtype='float')
        for i in range(K):
            for j in range(P):
                for d in range(M):
                    beta[i][j] += np.sum(phis[d][:,i]*(wws[d]==j).astype('int'))
            beta[i] /= np.sum(beta[i])

    return alpha, beta


# Perplexity calculation
def NLogLikelihood(ww, alpha, beta, num_iter=32):
    '''
    Function that calculates the negative log-likelihood of a document (or a set of words)
    with respect to an estimated LDA model
    Use Monte Carlo to estimate the log-likelihood.

    Parameters
    ----------
    ww : array-like, shape (N,)
        array of integer representing the set of words in the
    alpha : array-like, shape (K,)
        K being the number of topics in the model
        vector of topic mixing prior
    beta : array-like, shape (K, P)
        P being the number of words in the dictionary
        matrix of word distribution per topic
    num_iter : int, positive
        number of Monte Carlo iteration

    Return
    ------
    float, positive
        NLogLikelihood of the word list.
        The lower the nll is the better.
    '''

    # Initialization
    N = ww.shape[0]
    K, P = beta.shape
    acc_llh = 0.

    # Main loop
    for _ in range(num_iter):
        theta = np.random.dirichlet(alpha)
        llh = 1.
        for i in range(N):
            llh *= np.sum(theta * beta[:, ww[i]])
        acc_llh += llh

    return - np.log(acc_llh/num_iter)
    
def Perplexity(wws, alpha, beta, num_iter=32):
    '''
    Function that calculates the perplexity test dataset
        with respect to an estimated LDA model 
    Returns a positive number. The lower the perplexity is the better
        with random guessing the Perplexity should be P

    Parameters
    ----------
    ww : array-like, shape (N,)
        list of words with length N
    alpha : array-like, shape (K,)
        K being the number of topics in the model
        vector of topic mixing prior
    beta : array-like, shape (K, P)
        P being the number of words in the dictionary
        matrix of word distribution per topic
    num_iter : int, positive
        number of iteration 
    '''
    # Type and positivity check on num_iter
    if type(num_iter) != int: raise ValueError('num_iter has to be integer')
    if num_iter < 1: raise ValueError('num_iter has to be positive')

    # Initialization
    nom = 0.
    denom = 0

    # Main loop
    for ww in wws:
        nom += NLogLikelihood(ww, alpha, beta, num_iter)
        denom += ww.shape[0]

    return np.exp(nom/denom)


# Parameter estimation with calculating perplexity metric
def ParameterEstimationExtended(wws, K, P, holdouts, num_iter_MC = 32, num_iter_VI=10, num_iter_NR=3, num_iter_EM=10, printing = True):
    '''
    Parameter estimation algorithm
    with perplexity being calculated on holdout per each EM iteration

    Parameters
    ----------
    wws : array-like of array of integers
        list of vectors representing words
    K : int, positive
        number of topics
    P : int, positive
        number of words
    holdouts: array-like of array of array of integers
        list of holdout dataset
        each dataset is a list of documents
    num_iter_MC : int (default 32)
        numbef of MC iteration used for perplexity calculation
    num_iter_VI : int (default 10)
        number of VI iteration for algorithm 2
    num_iter_NR : int (default 3)
        number of Newton Raphson interation for algorithm 1
    num_iter_EM : int (default 10)
        number of EM iteration of the main loop
    printing : bool (default True)
        option to print out the loading bar using tqdm
    
    Return
    ------
    array-like of shape (K,)
        optimized estimation of alpha
    array-like of shape (K, P)
        optimized estimation of beta
    array-like of length of that of holdouts
        each element is a list of length num_iter_EM
        representing perplexity of that holdout dataset
        as a function of time.
    '''
    
    # Initialization
    M = len(wws)
    alpha = np.random.exponential(scale=2., size=K)
    beta = np.random.dirichlet([0.5]*P, size=K)
    perplexity = [[] for _ in range(len(holdouts))]

    # Variational EM main loop
    for _ in (tqdm.tqdm(range(num_iter_EM)) if printing else range(num_iter_EM)):

        # E step
        gammas = []
        phis = []
        for ww in wws:
            gamma, phi = VariationalInference(ww, alpha, beta)
            gammas.append(gamma)
            phis.append(phi)

        # M step for alpha
        for _ in range(num_iter_NR):
            g = M*(digamma(np.sum(alpha))-digamma(alpha))
            for m in range(M):
                g -= (digamma(np.sum(gammas[m]))-digamma(gammas[m]))
            h = - M * polygamma(1, alpha)
            z = M * polygamma(1, np.sum(alpha))
            alpha = np.maximum(1e-9, NewtonRaphson(alpha, g, h, z))

        # M step for beta
        beta = np.zeros_like(beta, dtype='float')
        for i in range(K):
            for j in range(P):
                for d in range(M):
                    beta[i][j] += np.sum(phis[d][:,i]*(wws[d]==j).astype('int'))
            beta[i] /= np.sum(beta[i])

        # Calculate Perplexity
        for i in range(len(holdouts)):
            perplexity[i].append(Perplexity(holdouts[i], alpha, beta, num_iter_MC))

    return alpha, beta, perplexity
    