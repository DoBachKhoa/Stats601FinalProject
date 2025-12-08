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
# ## Numerical Experiment 1: Parameter estimation for LDA

# %%
# Import libraries
import tqdm
import math
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.special import digamma, polygamma
inverse = lambda X: np.linalg.inv(X)

# %%
# Import implemented functions
import sys
sys.path.insert(1, '../algorithm')
from lda import VariationalInference, ParameterEstimation
from utils import discreteNormal, discretePoisson

# %%
# Data generation parameter for sub-experiment 1
np.random.seed(17)
K  = 3
P  = 10
M  = 100
N  = 100
words = list(range(10))
alpha = np.array([0.5, 0.7, 0.7])
dist0 = discreteNormal(P, 4.5, np.sqrt(1.))
dist1 = discretePoisson(P, 1., reverse = False)
dist2 = discretePoisson(P, 1., reverse = True)
beta  = np.vstack([dist0, dist1, dist2])

# %%
# Check parameter
print(alpha)
print(beta)

# %%
# Plot beta
fig, ax = plt.subplots(1, 4, figsize=(15, 3.2))
ax[0].bar(words, dist0)
ax[0].set_title('First distribution', fontsize=10)
ax[1].bar(words, dist1)
ax[1].set_title('Second distribution', fontsize=10)
ax[2].bar(words, dist2)
ax[2].set_title('Third distribution', fontsize=10)
ax[3].bar(words, alpha @ beta / np.sum(alpha))
ax[3].set_title('Mean mixed distribution', fontsize=10)
fig.suptitle(r'Plot of $\beta_i$s. $\alpha=$ ' + str(alpha), fontsize=10, y=1.08)
plt.show()

# %%
# Data generation for sub-experiment 1
dataset = []
for _ in range(M):
    theta = np.random.dirichlet(alpha) @ beta
    dataset.append(np.random.choice(P, N, p=theta, replace=True))

# %%
# Sneak peak
dataset[0]

# %%
np.random.seed(17)
alpha_hat, beta_hat = ParameterEstimation(dataset, K, P, num_iter_VI=10, num_iter_NR=3, num_iter_EM=50)

# %%
print(alpha_hat)
print(beta_hat)

# %%
# Plot beta_hat
fig, ax = plt.subplots(1, 4, figsize=(15, 3.2))
ax[0].bar(words, beta_hat[0])
ax[0].set_title('First distribution', fontsize=10)
ax[1].bar(words, beta_hat[1])
ax[1].set_title('Second distribution', fontsize=10)
ax[2].bar(words, beta_hat[2])
ax[2].set_title('Third distribution', fontsize=10)
ax[3].bar(words, alpha_hat @ beta_hat / np.sum(alpha))
ax[3].set_title('Mean mixed distribution', fontsize=10)
fig.suptitle(r'Plot of $\hat\beta_i$s. $\hat\alpha=$ ' + str(np.round(alpha_hat, 4)), fontsize=10, y=1.08)
plt.show()

# %%
# Repeat the experiment with a more complex dataset with 4 topics (sub-experiment 2)
np.random.seed(17)
K  = 4
P  = 10
M  = 100
N  = 100
alpha = np.array([0.5, 0.5, 0.8, 0.8])
dist0 = discreteNormal(P, 3.4, np.sqrt(1.))
dist1 = discreteNormal(P, 5.6, np.sqrt(1.))
dist2 = discretePoisson(P, 1., reverse = False)
dist3 = discretePoisson(P, 1., reverse = True)
beta  = np.vstack([dist0, dist1, dist2, dist3])

# %%
# Plot beta
fig, ax = plt.subplots(1, 5, figsize=(15, 2.5))
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
ax[4].bar(words, alpha @ beta / np.sum(alpha))
ax[4].set_title('Mean mixed distribution', fontsize=10)
fig.suptitle(r'Plot of $\beta_i$s. $\alpha=$ ' + str(alpha), fontsize=10, y=1.08)
plt.savefig('../../output/exp1_plot1.jpg', bbox_inches='tight', format='jpeg')
plt.show()

# %%
# Data generation for sub-experiment 2
dataset = []
for _ in range(M):
    theta = np.random.dirichlet(alpha) @ beta
    dataset.append(np.random.choice(P, N, p=theta, replace=True))

# %%
alpha_hat, beta_hat = ParameterEstimation(dataset, K, P, num_iter_VI=10, num_iter_NR=3, num_iter_EM=100)

# %%
print(alpha_hat)
print(beta_hat)

# %%
# Plot beta_hat
fig, ax = plt.subplots(1, 5, figsize=(15, 2.5))
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
ax[4].bar(words, alpha_hat @ beta_hat / np.sum(alpha))
ax[4].set_title('Mean mixed distribution', fontsize=10)
fig.suptitle(r'Plot of $\hat\beta_i$s. $\hat\alpha=$ ' + str(np.round(alpha_hat, 4)), fontsize=10, y=1.08)
plt.savefig('../../output/exp1_plot2.jpg', bbox_inches='tight', format='jpeg')
plt.show()

# %%
