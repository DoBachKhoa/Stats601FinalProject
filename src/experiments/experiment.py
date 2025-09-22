# Import libraries
import tqdm
import math
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix
from scipy.special import digamma, polygamma
inverse = lambda X: np.linalg.inv(X)

# Import implemented functions
import sys
sys.path.insert(1, 'src/algorithm')
from lda import VariationalInference, ParameterEstimation, ParameterEstimationExtended
from utils import discreteNormal, discretePoisson

def run_experiment_parameter_estimation(
    alpha,
    beta,
    seed = 17,
    M = 100,
    N = 100,
    num_iter_VI = 10,
    num_iter_NR = 3,
    num_iter_EM = 0,
    filename1 = 'fig_fitted_param.jpg',
    filename2 = 'fig_true_param.jpg',
    printing = True):
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
    filename1, filename2 : None, or str (default fig_fitted_param.jpg, fig_true_param.jpg)
        filename for two generated plots: respectively for the estimated parameters
        and the ground-truth parameters
        input None to not print
    printing : bool (default True)
        option to print the loading bar for the EM main loop using tqdm
    
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
    alpha_hat, beta_hat = ParameterEstimation(dataset, K, P, num_iter_VI=num_iter_VI,
                                              num_iter_NR=num_iter_NR, num_iter_EM=num_iter_EM, printing=printing)

    # Plot alpha_hat and beta_hat
    if filename1:
        fig, ax = plt.subplots(1, K+1, figsize=(15, 15/(K+1)+0.5))
        for k in range(K):
            ax[k].bar(words, beta_hat[k])
            ax[k].set_title('Distribution '+str(k), fontsize=10)
        ax[K].bar(words, alpha_hat @ beta_hat / np.sum(alpha))
        ax[K].set_title('Mean mixed distribution', fontsize=10)
        fig.suptitle(r'Plot of $\hat\beta_i$s. $\hat\alpha=$ ' + str(np.round(alpha_hat, 4)), fontsize=10, y=1.08)
        plt.savefig(filename1, bbox_inches='tight', format='jpeg')
        plt.close()

    # Plot alpha and beta
    if filename2:
        fig, ax = plt.subplots(1, K+1, figsize=(15, 15/(K+1)+0.5))
        for k in range(K):
            ax[k].bar(words, beta[k])
            ax[k].set_title('Distribution '+str(k), fontsize=10)
        ax[K].bar(words, alpha @ beta / np.sum(alpha))
        ax[K].set_title('Mean mixed distribution', fontsize=10)
        fig.suptitle(r'Plot of $\beta_i$s. $\alpha=$ ' + str(np.round(alpha, 4)), fontsize=10, y=1.08)
        plt.savefig(filename2, bbox_inches='tight', format='jpeg')
        plt.close()

    # Return fitted parameters
    return alpha_hat, beta_hat

def run_experiment_perplexity(
    alpha,
    beta,
    seed = 17,
    M = 100,
    N = 100,
    M1 = 10,
    N1 = 10,
    num_iter_VI = 10,
    num_iter_NR =3 ,
    num_iter_EM = 50,
    num_iter_MC = 200,
    filename = 'fig_perplexity.jpg',
    printing = True):
    '''
    Run parameter estimation experiment,
        with perplexity recorded and plotted.
    Plotting option is designed for K=4

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
    filename : None, or str (default fig_perplexity.jpg)
        filename for ground-truth parameters and perplexity
        input None to not print
    printing : bool (default True)
        option to print the loading bar for the EM main loop using tqdm
    
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
    if filename and K != 4: raise ValueError(f'Plotting option is designed for K=4 for athestic reasons')

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
    alpha_hat, beta_hat, perplexity = ParameterEstimationExtended(dataset, K, P, [dataset_test1, dataset_test2], num_iter_MC=400,
                                                                  num_iter_VI=10, num_iter_NR=3, num_iter_EM=num_iter_EM, printing=printing)
    
    # Plot alpha_hat and beta_hat
    if filename:

        # Create a figure and a gridspec layout
        fig = plt.figure(figsize=(14, 7))
        gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 2]) # Adjust width_ratios as needed

        # Add subplots using the gridspec layout
        ax1 = plt.subplot(gs[:, 2]) # Main plot spanning two rows
        ax2 = plt.subplot(gs[0, 0])
        ax3 = plt.subplot(gs[0, 1])
        ax4 = plt.subplot(gs[1, 0])
        ax5 = plt.subplot(gs[1, 1])

        # Plotting
        [dist0, dist1, dist2, dist3] = beta
        ax1.plot(perplexity[0], label='In-model data')
        ax1.plot(perplexity[1], label='Uniformly random data')
        ax1.hlines(y=P, xmin=0, xmax=num_iter_EM, linestyles = ['--'], label='Uniform random perplexity')
        ax1.set_title('Perplexity of LDA during training')
        ax1.set_xlabel('Number of EM iteration')
        ax1.legend()
        ax2.bar(words, dist0)
        ax2.set_title('First distribution', fontsize=10)
        ax3.bar(words, dist1)
        ax3.set_title('Second distribution', fontsize=10)
        ax4.bar(words, dist2)
        ax4.set_title('Third distribution', fontsize=10)
        ax5.bar(words, dist3)
        ax5.set_title('Forth distribution', fontsize=10)
        fig.suptitle(r'Plot of groundtruth parameters and perplexity. $\alpha=$ ' + str(alpha), fontsize=10)
        plt.savefig(filename, bbox_inches='tight', format='jpeg')
        plt.close()

    # Return fitted parameters
    return alpha_hat, beta_hat


if __name__ == '__main__':
    def experiment1():
        P=10
        words = list(range(10))
        alpha = np.array([0.5, 0.7, 0.7])
        dist0 = discreteNormal(P, 4.5, np.sqrt(1.))
        dist1 = discretePoisson(P, 1., reverse = False)
        dist2 = discretePoisson(P, 1., reverse = True)
        beta  = np.vstack([dist0, dist1, dist2])
        run_experiment_parameter_estimation(
            alpha,
            beta,
            seed = 17,
            M = 100,
            N = 100,
            num_iter_VI = 10,
            num_iter_NR = 3,
            num_iter_EM = 30,
            filename1 = 'output/fig_fitted_param.jpg',
            filename2 = 'output/fig_true_param.jpg',
            printing = True
        )

    def experiment2():
        alpha = np.array([0.75, 0.75, 0.75, 0.75])
        dist0 = [0.46, 0.46, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        dist1 = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.46, 0.46]
        dist2 = [0.01, 0.01, 0.01, 0.60, 0.32, 0.01, 0.01, 0.01, 0.01, 0.01]
        dist3 = [0.01, 0.01, 0.01, 0.01, 0.01, 0.32, 0.60, 0.01, 0.01, 0.01]
        beta  = np.vstack([dist0, dist1, dist2, dist3])
        run_experiment_perplexity(
            alpha,
            beta,
            seed = 17,
            M = 100,
            N = 100,
            M1 = 10,
            N1 = 10,
            num_iter_VI = 10,
            num_iter_NR =3 ,
            num_iter_EM = 50,
            num_iter_MC = 200,
            filename = 'output/fig_perplexity.jpg',
            printing = True
        )

    print('Runing first experiment ...')
    experiment1()

    print('Running second experiment ...')
    experiment2() # May take a bit