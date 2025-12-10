# Import libraries
import os
import tqdm
import math
import json
import random
import warnings
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix
from scipy.special import digamma, polygamma
inverse = lambda X: np.linalg.inv(X)

# Import implemented functions
from src.algorithm.ldaVI import VariationalInference, ParameterEstimation, ParameterEstimationExtended
from src.algorithm.utils import discreteNormal, discretePoisson, discreteInput

def run_experiment_parameter_estimation(
    alpha,
    beta,
    seed = 17,
    M = 100,
    N = 100,
    num_iter_VI = 10,
    num_iter_NR = 3,
    num_iter_EM = 0,
    printing = True,
    baseline=False):
    '''
    Run parameter estimation experiment

    Parameters
    ----------
    alpha : array-like of shape (K)
        ground-truth dirichlet distribution parameter for topics
    beta : array-like of shape (K, P)
        ground-truth distribution parameter for vocabulary per topic
    seed : int (default 17)
        random seed
    M : int (default 100)
        number of generated documents
    N : int (default 100)
        number of generated word per documents
    num_iter_VI : int (default 10)
        number of Variational Inference iteration
    num_iter_NR : int (default 3)
        number of Newton Raphson iteration
    num_iter_EM : int (default 50)
        number of Variation EM iteration for the main algorithm
    printing : bool (default True)
        option to print the loading bar for the EM main loop using tqdm
    baseline : bool (default False)
        use the baseline unvectorized version
    
    Return
    ------
    array-like, shape (K)
        estimation for alpha
    array-like, shape (K, P)
        estimation for beta
    '''

    # Data generation
    K, P = beta.shape
    words = list(range(P))
    np.random.seed(seed)
    dataset = []
    for _ in range(M):
        theta = np.random.dirichlet(alpha) @ beta
        dataset.append(np.random.choice(P, N, p=theta, replace=True))

    # Run parameter estimation algorithm
    np.random.seed(seed)
    alpha_hat, beta_hat = ParameterEstimation(dataset, K, P, num_iter_VI=num_iter_VI, baseline=baseline,
                                              num_iter_NR=num_iter_NR, num_iter_EM=num_iter_EM, printing=printing)

    # Return fitted parameters
    return alpha_hat, beta_hat

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-r', help='Experiment randomization seed (default: 17)', 
                        action='store', default=17, type=int)
    parser.add_argument('--num_doc', '-m', help='Number of documents (default 100)', action='store', default=100, type=int)
    parser.add_argument('--num_word', '-n', help='Number of words per document (default 100)', action='store', default=100, type=int)
    parser.add_argument('--iter_VI', '-v', help='Number of VI iterations (default 10)', action='store', default=10, type=int)
    parser.add_argument('--iter_NR', '-q', help='Number of Newton Raphson iterations (default 3)', action='store', default=3, type=int)
    parser.add_argument('--iter_EM', '-e', help='Number of EM algorithm iterations (default 30)', action='store', default=30, type=int)
    parser.add_argument('--param_file', '-i', help='Input json file for ground truth parameters (default params/params1.json)', 
                        action='store', default='params/params1.json')
    parser.add_argument('--output_file', '-o', help='Output plot file', action='store', default='output/fitted_parameters.jpg')
    parser.add_argument('--silent', '-s', help='Option to run without printing', action='store_true')
    parser.add_argument('--unvectorized', '-u', help='Use baseline unvectorized version', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':

    # Parsing arguments
    args = parse_arguments()
    with open(args.param_file, 'r') as file:
        params = json.load(file)
    P = params['P']
    K = len(params['alpha'])
    words = list(range(P))

    # Formating alpha & beta
    discrete_dict = {
        'gaussian' : discreteNormal,
        'normal' : discreteNormal,
        'poisson' : discretePoisson,
        'input' : discreteInput
    }
    alpha = np.array(params['alpha'], dtype='float')
    beta_raw = []
    for type, param in zip(params['beta_type'], params['beta_param']):
        beta_raw.append(discrete_dict[type](**param))
    beta = np.vstack(beta_raw)
    
    # Printing hyper-parameters
    if not args.silent:
        print(f'======== Running experiment 1 ========')
        print(f"==  Seed: {args.seed}{' '*(26-len(str(args.seed)))}==")
        print(f"==  M: {args.num_doc}{' '*(29-len(str(args.num_doc)))}==")
        print(f"==  N: {args.num_word}{' '*(29-len(str(args.num_word)))}==")
        print(f"==  Num. iter. VI: {args.iter_VI}{' '*(17-len(str(args.iter_VI)))}==")
        print(f"==  Num. iter. NR: {args.iter_NR}{' '*(17-len(str(args.iter_NR)))}==")
        print(f"==  Num. iter. EM: {args.iter_EM}{' '*(17-len(str(args.iter_EM)))}==")
        print(f"==  Baseline (Unvec) ver.: {args.unvectorized}{' '*(9-len(str(args.unvectorized)))}==")
        print(f'======================================')

    # Running experiment
    alpha_hat, beta_hat = run_experiment_parameter_estimation(
        alpha,
        beta,
        seed = args.seed,
        M = args.num_doc,
        N = args.num_word,
        num_iter_VI = args.iter_VI,
        num_iter_NR = args.iter_NR,
        num_iter_EM = args.iter_EM,
        printing = 1-args.silent,
        baseline = args.unvectorized
    )

    # Plot alpha_hat and beta_hat
    y_max = max(np.max(beta), np.max(beta_hat))
    fig, ax = plt.subplots(2, K+1, figsize=(3*K+3.2, 6.2), constrained_layout=True)
    ax[0][0].set_ylabel('True parameter $\\beta$', fontweight='bold')
    ax[1][0].set_ylabel('Fitted parameter $\\hat\\beta$', fontweight='bold')
    for k in range(K):
        ax[0][k].set_title('Distribution '+str(k), fontweight='bold')
        ax[0][k].bar(words, beta[k])
        ax[1][k].bar(words, beta_hat[k])
        ax[0][k].set_ylim(bottom=0., top=y_max*1.1)
        ax[1][k].set_ylim(bottom=0., top=y_max*1.1)
        ax[0][k].set_xticklabels([])
        ax[1][k].set_xticklabels([])
        if k != 0:
            ax[0][k].set_yticklabels([])
            ax[1][k].set_yticklabels([])
    ax[0][K].set_title('Mean mixed distribution', fontweight='bold')
    ax[0][K].bar(words, alpha @ beta / np.sum(alpha))
    ax[1][K].bar(words, alpha_hat @ beta_hat / np.sum(alpha))
    ax[0][K].set_ylim(bottom=0., top=y_max*1.1)
    ax[1][K].set_ylim(bottom=0., top=y_max*1.1)
    ax[0][K].set_yticklabels([])
    ax[1][K].set_yticklabels([])
    ax[0][K].set_xticklabels([])
    ax[1][K].set_xticklabels([])
    
    # Title & saving
    fig.suptitle(f'Plot of fitted parameters. $\\alpha={str(np.round(alpha, 4))},$ ' + \
                    f'$\\hat\\alpha={str(np.round(alpha_hat, 4))}$', fontweight='bold', y=1.08)
    plt.savefig(args.output_file, bbox_inches='tight', format='jpeg')
    plt.close()
