

# Bayesian Mixture Models.


[![Build Status](https://travis-ci.org/chariff/BayesianMixtures.svg?branch=master)](https://travis-ci.org/chariff/BayesianMixtures)
[![Codecov](https://codecov.io/github/chariff/BayesianMixtures/badge.svg?branch=master&service=github)](https://codecov.io/github/chariff/BayesianMixtures?branch=master)
[![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT)



Python package to perform Dirichlet process mixture model clustering.  
The Dirichlet process prior on the mixing distribution allows for an inference of the number of classes directly 
from the data thus avoiding model selection issues. Gaussian mixture models or Skew-t mixture models are available. Skew-t distributions provide robustness to outliers and non-elliptical shape of clusters. Theoretically, Gaussian distributions are nested within Skew-t distributions but in practice this increase of flexibility comes at the cost of a more approximate inference. Inference is done using a collapsed gibbs sampling scheme. 

Installation.
============

### Installation

* From GitHub:

      pip install git+https://github.com/chariff/BayesianMixtures.git

### Dependencies
BayesianMixtures requires:
* Python (>= 3.5)
* NumPy (>= 1.18.5)
* SciPy (>= 1.4.1)
* sklearn (>= 0.23.2)


Brief guide to using BayesianMixtures.
=========================

Checkout the package docstrings for more information.

## 1. Fitting an infinite Gaussian mixture model and making predictions.

```python
from BayesianMixtures.gaussian_mixture import BayesianGaussianMixture
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from itertools import cycle
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

```
Simulated data.

```python

n_samples = 2500
n_components = 4
dim = 2

data_samples, labels = make_blobs(n_samples=n_samples, n_features=2,
                                  centers=4, cluster_std=1, random_state=0)

```

Select the number maximum number of mcmc (Monte Carlo Markov Chain) iterations.
 
```python
# Maximum number of mcmc iterations 
max_iter = 1000
# Burn-in iterations
burn_in = 500
```
Instantiate a BayesianGaussianMixture object with default parameters.
```python
cls = BayesianGaussianMixture(max_iter=max_iter, burn_in=burn_in  max_components=100,
                           verbose=2, verbose_interval=500,  n_components_init=100,
                           random_state=2026, init_params='kmeans')
```
Perform inference. 
```python
p = cls.fit(data_samples)
```
Maximum a posteriori (MAP) partition.
```python
map_partition = p.map_partition
```
```python
with plt.style.context('bmh'):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax1, ax2 = axes.ravel()
    alpha = .4
    # Scatter plot of the MAP partition
    for label in set(labels):
        ax1.scatter(data_samples[labels == label, 0],
                    data_samples[labels == label, 1], 
                    c=next(color_cycle), alpha=alpha, zorder=2)
        
        ax1.set_title('Actual')
        ax1.set_xlabel('feature 1')
        ax1.set_ylabel('feature 2')

    for label in set(map_partition):
        ax2.scatter(data_samples[map_partition == label, 0],
                    data_samples[map_partition == label, 1], 
                    c=next(color_cycle), alpha=alpha, zorder=2)
        
        ax2.set_title('MAP partition')
        ax2.set_xlabel('feature 1')
        ax2.set_ylabel('feature 2')
    plt.show()
```
![MAP partition](https://github.com/chariff/BayesianMixtures/raw/master/examples/MAP_partition_0.png)

Log posterior trace.
```python
plt.plot(p.logposteriors,  linewidth=.6, c='black')
plt.title('Log posterior trace')
plt.xlabel('mcmc iterations')
plt.ylabel('Log posterior evaluation')
plt.show()
```
![Log posterior trace](https://github.com/chariff/BayesianMixtures/raw/master/examples/trace_0.png)

Predict new values.
```python
# predict data_samples for the sake of example
map_predicted_partition = p.map_predict(data_samples)
```

Using the MAP approach, clustering uncertainty cannot be assessed.
In the following section we will show an example of how to use the sampled partitions to assess
the clustering uncertainty.


The following function is to calculate an average of the co-clustering
matrices from the sampled partitions to obtain the posterior co-clustering probabilities.
```python
from numba import jit

@jit(nopython=True)
def coclustering(partitions):
    n_mcmc_samples, n_samples = partitions.shape
    co_clustering = np.zeros(shape=(n_samples, n_samples))
    for partition in partitions:
        for i in range(n_samples):
            label_obs = partition[i]
            for j in range(n_samples):
                if partition[j] == label_obs:
                    co_clustering[i, j] += 1
    co_clustering /= n_mcmc_samples
    return co_clustering
```
```python
# explored partitions in the posterior mcmc draws
partitions = p.partitions
# compute the average co-clustering matrix
coclust = coclustering(partitions)
```
Heatmap of the posterior co-clustering probabilities.
```python
# Plot co-clustering matrix.
import seaborn as sns
g = sns.clustermap(coclust, metric='euclidean', method='average', cmap='viridis', 
                   col_cluster=True, row_cluster=True,  dendrogram_ratio=0.1, 
                   xticklabels=False, yticklabels=False, figsize=(7, 7))
ax = g.ax_heatmap
ax.set_title("Average co-clustering matrix")
ax.set_xlabel("Observations")
ax.set_ylabel("Observations")

g.ax_cbar.set_position((0.01, 0.4, .05, .2))
g.ax_row_dendrogram.set_visible(False)
g.ax_col_dendrogram.set_visible(False)
plt.show()
```
![Co-clustering probabilities](https://github.com/chariff/BayesianMixtures/raw/master/examples/avg_coclust.png)

To obtain a point estimate of the clustering, one
could minimize a loss function of the co-clustering matrices from the 
sampled partitions and the co-clustering probabilities.

### References:
* https://academic.oup.com/biostatistics/article/11/2/317/268224
* https://projecteuclid.org/euclid.aoas/1554861663
* https://projecteuclid.org/euclid.aos/1056562461
* https://www.tandfonline.com/doi/abs/10.1080/03610910601096262
* https://link.springer.com/article/10.1007/s11222-009-9150-y
* https://www.stat.berkeley.edu/~pitman/621.pdf
* https://www.jstor.org/stable/24305538?seq=1
* http://www2.stat.duke.edu/~mw/MWextrapubs/West1992alphaDP.pdf


    -- Chariff Alkhassim