# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Numerical Experiment 2: Perplexity as a metric for generalization

# %%
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

# %%
# Import implemented functions
import sys
sys.path.insert(1, '../algorithm')
from lda import VariationalInference, ParameterEstimation, ParameterEstimationExtended
from utils import discreteNormal, discretePoisson

# %%
# Data generation with 4 topics
np.random.seed(17)
K  = 4
P  = 10
M  = 100
N  = 100
M1 = 10
N1 = 10
words = list(range(10))
alpha = np.array([1., 1., 1.5, 1.5])
dist0 = discreteNormal(P, 3.4, np.sqrt(1.))
dist1 = discreteNormal(P, 5.6, np.sqrt(1.))
dist2 = discretePoisson(P, 1., reverse = False)
dist3 = discretePoisson(P, 1., reverse = True)
beta  = np.vstack([dist0, dist1, dist2, dist3])

# %%
# Ploting of parameters
fig, ax = plt.subplots(1, 5, figsize=(15, 4))
for a in ax.flat:
    a.set_ylim(top=0.54)
ax[0].bar(words, dist0)
ax[0].set_title('First distribution', fontsize=10)
ax[1].bar(words, dist1)
ax[1].set_title('Second distribution', fontsize=10)
ax[2].bar(words, dist2)
ax[2].set_title('Third distribution', fontsize=10)
ax[3].bar(words, dist3)
ax[3].set_title('Forth distribution', fontsize=10)
ax[4].bar(words, alpha @ beta / sum(alpha))
ax[4].set_title('Mean mixed distribution', fontsize=10)
fig.suptitle(r'Plot of $\beta_i$s. $\alpha=$ ' + str(alpha), fontsize=10)
plt.show()

# %%
# Data generation
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

# %%
num_iter_EM = 100
alpha_hat, beta_hat, perplexity = ParameterEstimationExtended(dataset, K, P, [dataset_test1, dataset_test2], \
                                                              num_iter_MC=400, num_iter_VI=10, num_iter_NR=3, num_iter_EM=num_iter_EM)

# %%
plt.plot(perplexity[0], label='In-model data')
plt.plot(perplexity[1], label='Uniformly random data')
plt.hlines(y=P, xmin=0, xmax=num_iter_EM, linestyles = ['--'], label='Uniform random perplexity')
plt.title('Perplexity of LDA during training')
plt.xlabel('Number of EM iteration')
plt.legend()
plt.show()

# %%
# Plotting for the report

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
# fig.suptitle(r'Plot of groundtruth parameters and perplexity. $\alpha=$ ' + str(alpha), fontsize=10)
plt.savefig('../../output/exp2_plot1.jpg', bbox_inches='tight', format='jpeg')
plt.show()

# %%
# perplexity[0][-5:]

# %%
perplexity[1][-5:]

# %%
# Plot beta_hat - how does the fitting look?
fig, ax = plt.subplots(1, 5, figsize=(15, 4))
for a in ax.flat:
    a.set_ylim(top=0.54)
ax[0].bar(words, beta_hat[0])
ax[0].set_title('First distribution', fontsize=10)
ax[1].bar(words, beta_hat[1])
ax[1].set_title('Second distribution', fontsize=10)
ax[2].bar(words, beta_hat[2])
ax[2].set_title('Third distribution', fontsize=10)
ax[3].bar(words, beta_hat[3])
ax[3].set_title('Forth distribution', fontsize=10)
ax[4].bar(words, alpha_hat @ beta_hat / sum(alpha))
ax[4].set_title('Mean mixed distribution', fontsize=10)
fig.suptitle(r'Plot of $\hat\beta_i$s. $\hat\alpha=$ ' + str(np.round(alpha_hat, 4)), fontsize=10)
plt.show()

# %%
# Plot of final perplexity as a function of topic
K_max = 7
num_iter = 4
perplexities = []
for k in tqdm.tqdm(range(1, K_max+1)):
    temp = 0.
    for _ in range(num_iter):
        _, _, perplexity = ParameterEstimationExtended(dataset, K, P, [dataset_test1], printing = False, \
                                                       num_iter_MC=400, num_iter_VI=10, num_iter_NR=3, num_iter_EM=25)
        temp += perplexity[-1][-1]
    temp /= num_iter
    perplexities.append(temp)


# %%
plt.plot(list(range(1, K_max+1)), perplexities)
plt.title('Perplexity as a function of number of topics')
plt.xlabel('Number of topics')
plt.savefig('../../output/exp2_plot3.jpg', format='jpeg')
plt.show()

# %%
# Data generation with 4 topics - attempt 2
np.random.seed(17)
K  = 4
P  = 10
M  = 100
N  = 100
M1 = 10
N1 = 10
alpha = np.array([0.75, 0.75, 0.75, 0.75])
dist0 = [0.46, 0.46, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
dist1 = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.46, 0.46]
dist2 = [0.01, 0.01, 0.01, 0.60, 0.32, 0.01, 0.01, 0.01, 0.01, 0.01]
dist3 = [0.01, 0.01, 0.01, 0.01, 0.01, 0.32, 0.60, 0.01, 0.01, 0.01]
beta  = np.vstack([dist0, dist1, dist2, dist3])

# %%
# Ploting of parameters
fig, ax = plt.subplots(1, 5, figsize=(15, 4))
ax[0].plot(dist0)
ax[0].set_title('First distribution', fontsize=10)
ax[1].plot(dist1)
ax[1].set_title('Second distribution', fontsize=10)
ax[2].plot(dist2)
ax[2].set_title('Third distribution', fontsize=10)
ax[3].plot(dist3)
ax[3].set_title('Forth distribution', fontsize=10)
ax[4].plot(alpha @ beta / sum(alpha))
ax[4].set_title('Mean mixed distribution', fontsize=10)
fig.suptitle(r'Plot of $\beta_i$s. $\alpha=$ ' + str(alpha), fontsize=10)
plt.show()

# %%
# Data generation
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

# %%
num_iter_EM = 100
alpha_hat, beta_hat, perplexity = ParameterEstimationExtended(dataset, K, P, [dataset_test1, dataset_test2], \
                                                              num_iter_MC=400, num_iter_VI=10, num_iter_NR=3, num_iter_EM=num_iter_EM)

# %%
# Plotting for the report

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
# fig.suptitle(r'Plot of groundtruth parameters and perplexity. $\alpha=$ ' + str(alpha), fontsize=10)
plt.savefig('../../output/exp2_plot2.jpg', bbox_inches='tight', format='jpeg')
plt.show()

# %%
perplexity[0][-5:]

# %%
perplexity[1][-5:]

# %%
perplexity[0][:5]

# %%
perplexity[1][:5]

# %%
