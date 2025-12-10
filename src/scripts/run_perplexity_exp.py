# Import libraries
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

def run_experiment_perplexity(
    alpha,
    beta,
    seed = 17,
    M = 100,
    N = 100,
    M1 = 10,
    N1 = 10,
    num_iter_VI = 10,
    num_iter_NR = 3,
    num_iter_EM = 50,
    num_iter_MC = 200,
    printing = True,
    baseline=False):
    '''
    Run parameter estimation experiment, with perplexity recorded.

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
    M1 : int (default 10)
        number of generated documents for holdout documents
    N1 : int (default 10)
        number of generated word per holdout document
    num_iter_VI : int (default 10)
        number of Variational Inference iteration
    num_iter_NR : int (default 3)
        number of Newton Raphson iteration
    num_iter_EM : int (default 50)
        number of Variation EM iteration for the main algorithm
    num_iter_MC : int (default 200)
        number of Monte Carlo iteration used to estimate perplexity
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
    array-like, shape 
    '''
    # Comparability check
    K, P = beta.shape
    words = list(range(P))

    # Data generation
    np.random.seed(seed)
    dataset = []
    for _ in range(M):
        theta = np.random.dirichlet(alpha) @ beta
        dataset.append(np.random.choice(P, N, p=theta, replace=True))

    # Also generating holdout data
    dataset_test1 = []
    for _ in range(M1):
        theta = np.random.dirichlet(alpha) @ beta
        dataset_test1.append(np.random.choice(P, N1, p=theta, replace=True))

    # Generating data from uniformly random choice model
    dataset_test2 = []
    for _ in range(M1):
        dataset_test2.append(np.random.choice(P, N1, replace=True))

    # Run parameter estimation algorithm
    np.random.seed(seed)
    alpha_hat, beta_hat, perplexity = ParameterEstimationExtended(dataset, K, P, [dataset_test1, dataset_test2], num_iter_MC=num_iter_MC,
                                                                  num_iter_VI=num_iter_VI, num_iter_NR=num_iter_NR,
                                                                  num_iter_EM=num_iter_EM, printing=printing, baseline=baseline)

    # Return fitted parameters & perplexity
    return alpha_hat, beta_hat, perplexity

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-r', help='Experiment randomization seed (default: 17)', 
                        action='store', default=17, type=int)
    parser.add_argument('--num_doc', '-m', help='Number of documents (default 100)', action='store', default=100, type=int)
    parser.add_argument('--num_word', '-n', help='Number of words per document (default 100)', action='store', default=100, type=int)
    parser.add_argument('--iter_VI', '-v', help='Number of VI iterations (default 10)', action='store', default=10, type=int)
    parser.add_argument('--iter_NR', '-q', help='Number of Newton Raphson iterations (default 3)', action='store', default=3, type=int)
    parser.add_argument('--iter_EM', '-e', help='Number of EM algorithm iterations (default 30)', action='store', default=30, type=int)
    parser.add_argument('--iter_MC', '-c', help='Number of MC iterations (default 200)', action='store', default=200, type=int)
    parser.add_argument('--param_file', '-i', help='Input json file for ground truth parameters (default params/params2.json)', 
                        action='store', default='params/params2.json')
    parser.add_argument('--output_file', '-o', help='Output plot file', action='store', default='output/plot_perplexity.jpg')
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
        print(f'======== Running experiment 2 ========')
        print(f"==  Seed: {args.seed}{' '*(26-len(str(args.seed)))}==")
        print(f"==  M: {args.num_doc}{' '*(29-len(str(args.num_doc)))}==")
        print(f"==  N: {args.num_word}{' '*(29-len(str(args.num_word)))}==")
        print(f"==  Num. iter. VI: {args.iter_VI}{' '*(17-len(str(args.iter_VI)))}==")
        print(f"==  Num. iter. NR: {args.iter_NR}{' '*(17-len(str(args.iter_NR)))}==")
        print(f"==  Num. iter. EM: {args.iter_EM}{' '*(17-len(str(args.iter_EM)))}==")
        print(f"==  Num. iter. MC: {args.iter_MC}{' '*(17-len(str(args.iter_MC)))}==")
        print(f"==  Baseline (Unvec) ver.: {args.unvectorized}{' '*(9-len(str(args.unvectorized)))}==")
        print(f'======================================')

    # Running experiment
    alpha_hat, beta_hat, perplexity = run_experiment_perplexity(
        alpha,
        beta,
        seed = args.seed,
        M = args.num_doc,
        N = args.num_word,
        num_iter_VI = args.iter_VI,
        num_iter_NR = args.iter_NR,
        num_iter_EM = args.iter_EM,
        num_iter_MC = args.iter_MC,
        printing = 1-args.silent,
        baseline = args.unvectorized
    )

    # Create a figure and a gridspec layout
    K1 = (K+1)//2
    fig = plt.figure(figsize=(3*(K1+2)+0.2, 6.2))
    gs = gridspec.GridSpec(2, K1+1, width_ratios=[1]*K1+[2])

    # Add subplots using the gridspec layout
    ax1 = plt.subplot(gs[:, -1]) # Main plot spanning two rows
    axes = [plt.subplot(gs[i//K1, i-i//K1*K1]) for i in range(K)]

    # Plotting
    ax1.plot(perplexity[0], label='In-model data')
    ax1.plot(perplexity[1], label='Uniformly random data')
    ax1.hlines(y=P, xmin=0, xmax=args.iter_EM, linestyles = ['--'], label='Uniform random perplexity')
    ax1.set_title('Perplexity of LDA during training')
    ax1.set_xlabel('Number of EM iteration')
    ax1.legend()
    for i in range(K):
        axes[i].bar(words, beta[i, :])
        axes[i].set_title(f'Distribution {i}', fontsize=10)
        axes[i].set_xticklabels([])
    fig.suptitle(r'Plot of groundtruth parameters and perplexity. $\alpha=$ ' + str(alpha), fontsize=10)
    plt.savefig(args.output_file, bbox_inches='tight', format='jpeg')
    plt.close()
