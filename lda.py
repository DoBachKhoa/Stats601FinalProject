import tqdm
import math
import numpy as np
from scipy.special import digamma, polygamma


# Algorithm 1: Newton Raphson update
def NewtonRaphson(alpha, g, h, z):
    '''
    Newton Rahpson update in linear time
    for Hessian matrix with special structure
    Algorithm 1 in the report
    '''
    # Calculates c in linear time
    c = np.sum(g/h) / (1/z + np.sum(1./h))

    # Returns new alpha in linear time
    return alpha - ((g-c)/h)


# Algorithm 2: Variational inference
def VariationalInference(ww, alpha, beta, num_iter=10):
    '''
    Variational Inference algorithm
    ww is the list of words
    alpha is a K-vector of topic mixing prior
    beta is a K-by-P matrix of word distribution per topic
    num_iter: number of coordinate ascent loop
    Algorithm 2 in the report
    '''
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
    Parameter estimation algorithm
    wws is a list of vectors representing words
    K is the number of topics
    P is the number of words
    the rest are hyper-parameters for numbers of iterations
    returns fitted value of alpha and beta
    Algorithm 3 in the report
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

    return alpha, beta


# Perplexity calculation
def NLogLikelihood(ww, alpha, beta, num_iter=32):
    '''
    Function that calculates the negative log-likelihood of a document (or a set of words)
    with respect to an estimated LDA model
    Returns a positive number. The lower the nll is the better
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
    '''
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
    