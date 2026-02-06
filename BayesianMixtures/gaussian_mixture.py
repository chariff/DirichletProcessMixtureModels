# Author: Chariff Alkhassim <chariff.alkhassim@gmail.com>
# License: MIT

import sys

import numpy as np

from scipy.stats import (beta, gamma, norm, multivariate_normal,
                         uniform, invwishart)
from scipy.special import (loggamma, polygamma)
from scipy.linalg import eigvalsh


from .gibbs_sampling_base import GibbsSamplingMixture

from collections import Counter

from sklearn.utils import check_array


_MACHINE_FLOAT_MIN = sys.float_info.min
_RANDOM_STATE_UPPER = 1e5



def _check_positive_scalar(x, name):
    """Check if a value x is a positive scalar."""

    if not x > 0 and np.isscalar(x):
        raise ValueError("The parameter '%s' should be a positive scalar, "
                         "but got %s" % (name, x))


def _clusters_sizes(labels, max_components):
    """Compute the number of components and the size of
       samples in each component given an array of labels.

    Parameters
    ----------
    labels : array of shape (n_observations, )
        The labels of each observation.

    max_components : int
        Maximum number of components.

    Returns
    -------
    components_sizes : array of shape (max_components, )
        The size of samples in each component.

    n_components : int
        The number of components.

    components_labels : list, (n_components)
        The labels of each component.
    """

    components_sizes = np.zeros(max_components, dtype=int)
    table_labels = Counter(labels)
    n_components = len(table_labels)
    components_labels = list(table_labels.keys())
    for key, value in table_labels.items():
        components_sizes[key] = value

    return components_sizes, n_components, components_labels


def log_jeffrey_prior(x):
    """ Non-informative prior distribution for a parameter space

    Parameters
    ----------
    x : positive scalar

    Returns
    -------
    Logarithm transformation of the Jeffrey prior : float
    """

    p1 = polygamma(n=1, x=x / 2)
    p2 = polygamma(n=1, x=(x + 1) / 2) + 2 * (x + 3) / (x * (x + 1) ** 2)
    if p1 <= p2:
        return -1 * float('inf')
    return .5 * (np.log(x) - np.log(x + 3) + np.log(p1 - p2))


def _niw_posterior_params(X, 
                          loc_prior, 
                          scaling_prior, 
                          scale_prior,
                          degrees_of_freedom_prior):
    """Normal-inverse-Wishart posterior parameters estimation.

    Parameters
    ----------

    X : array of shape (n_observations, n_features)
        Observations modelled by a gaussian distribution.

    loc_prior :  array of shape (n_features, )
        Location hyper parameter.

    scaling_prior :  float
        Scaling hyper parameter.

    scale_prior : array of shape (n_features, n_features)
        Scale hyper parameter.

    degrees_of_freedom_prior : float
        Degree of freedom hyper parameter. Must be superior to n_features - 1.


    n_components : int
        Number of components of the gaussian mixture.

    n_features : int
        Number of dimensions of the gaussian mixture.

    allow_singular : boolean
        If a matrix to be inverted happens to be singular, the pseudo-inverse
        is computed instead.

    Returns
    -------

    loc : array of shape (n_features, )

    scaling : float

    scale : array of shape (n_features, n_features)

    degree_of_freedom : float

    References
    ----------

    """
    n = len(X)

    # posterior degrees of freedom
    degrees_of_freedom = degrees_of_freedom_prior + n

    # posterior scaling
    scaling = scaling_prior + n 

    # posterior location
    mu = X.mean(axis=0) 
    loc = (scaling_prior * loc_prior + n * mu) / (scaling_prior + n)

    # posterior scale
    # TODO hierarchical Wishart distribution prior on the scale prior.
    X_centered = X - mu
    scale_part1 = (X_centered[:, :, None] @ X_centered[:, None]).sum(axis=0)
    mu_centered = mu - loc_prior
    scale_part2 = (scaling_prior * n / (scaling_prior + n)) * \
        (mu_centered.T @ mu_centered) 
    scale = scale_prior + scale_part1 + scale_part2

    
    return loc, scaling, scale, degrees_of_freedom


def _niw_rvs(loc, scaling, scale, degrees_of_freedom, random_state=None):
    """Sample from the Normal-inverse-Wishart distribution.

    Parameters
    ----------

    loc : array of shape (n_features, )
        Location parameter.

    scaling : float
        Scaling parameter.

    scale : array of shape (n_features, n_features)
        Scale parameter.

    degrees_of_freedom : float
        Degree of freedom parameter. Must be superior to n_features - 1.

    dim : int
        Dimension of the parameter space. Equal to n_features.

    random_state : boolean, default=None
        Optional, random seed used to initialize the pseudorandom number generator.
        If the random seed is None the np.random.RandomState singleton is used.


    Returns
    -------

    loc : array of shape (n_features, )

    scale : array of shape (n_features, n_features)
    """
    
    scale = invwishart.\
        rvs(df=degrees_of_freedom, scale=scale, random_state=random_state)
    loc = multivariate_normal.\
        rvs(mean=loc, cov=scale / scaling, random_state=random_state)

    return loc, scale


def _niw_logpdf(loc_obs, scale_obs, loc, scaling, scale,
                            degrees_of_freedom, allow_singular):
    """Normal-inverse-Wishart log pdf.

    Parameters
    ----------

    loc_obs : array of shape (n_features, )
        Location observation.

    scale_obs : array of shape (n_features, n_features)
        Scale observation.

    loc : array of shape (n_features, )
        Location parameter.

    scaling :  float
        Scaling parameter.

    scale : array of shape (n_features, n_features)
        Scale parameter.

    degrees_of_freedom : float
        Degree of freedom parameter. Must be superior to n_features - 1.

    allow_singular : boolean
        If a matrix to be inverted happens to be singular, the pseudo-inverse
        is computed instead.


    Returns
    -------

    eval : float
        Evaluation of the pdf at the observations conditionally on parameters.
    """
    inv_w_logpdf = invwishart.\
        logpdf(scale_obs, df=degrees_of_freedom, scale=scale)
    normal_logpdf = multivariate_normal.\
        logpdf(loc_obs, mean=loc, cov=scale_obs / scaling, 
               allow_singular=allow_singular)

    return inv_w_logpdf + normal_logpdf


class BayesianGaussianMixture(GibbsSamplingMixture):
    """Bayesian estimation of a gaussian mixture using a collapsed
    gibbs sampling scheme.

    This class allows to infer an approximate posterior distribution over
    the parameters of a gaussian mixture distribution. The effective number
    of components can be inferred from the data. The type of prior for the
    weights distribution implemented in this class is an infinite mixture
    model with the Dirichlet Process (using the Stick-breaking representation).
    A slice sampler for the Stick breaking approach to the Dirichlet process
    elegantly enables a finite number of centers to be sampled within
    each iteration of the MCMC.


    .. versionadded:: 0.1

    Parameters
    ----------

    max_iter : int, default=100
        The number of MCMC iterations to perform. The number of
        samplings will be equal to max_iter - burn_in.

    burn_in : int, default=20
        The number of MCMC burn-in iterations to perform.

    n_components_init : int
        Initial number of components.

    max_components : int
        Maximum number of components.

    init_params : {'kmeans', 'random'}, default='kmeans'
        The method used to initialize the weights, the means and the
        covariances.
        Must be one of::
            'kmeans' : responsibilities are initialized using kmeans.
            'random' : responsibilities are initialized randomly.

    loc_prior : float or array of shape (n_features,), default=None.
        The hyper parameter of the Gaussian prior on the location distribution.
        If it is None, it is set to the mean of X.

    scaling_prior : float, default=None.
        The hyper parameter scaling the scale of the Gaussian prior.
        If it is None, it's set to `n_features`.

    scale_prior : float or array of shape (n_features, n_features), default=None.
        The hyper parameter of the inverse-Wishart prior on the scale distribution.
        If it is None, the emiprical covariance prior is initialized using the
        covariance of X.

    degrees_of_freedom_prior : float, default=None.
        The hyper parameter degrees of freedom of the inverse-Wishart prior on
        the scale distribution.
        If it is None, it's set to `n_features`.


    alpha_a_prior : float, default=1e-3
        The hyper parameter shape of the gamma prior on the scale factor
        (alpha) in the Dirichlet process.

    alpha_b_prior : float, default=1e-3
        The hyper parameter inverse scale of the gamma prior on the scale factor
        alpha in the Dirichlet process.

    allow_singular : boolean
        If a matrix to be inverted happens to be singular, the pseudo-inverse
        is computed instead.

    random_state : boolean, default=None
        Optional, random seed used to initialize the pseudorandom number generator.
        If the random seed is None np.random.randint is used to generate a seed.

    verbose : int, default=0
        Enable verbose output. If 1 then it prints each iteration step.
        If greater than 1 then it prints also the log posterior and the time
        needed for each step.

    verbose_interval : int, default=10
        Number of iteration done before the next print.

    Attributes
    ----------
    TODO

    Examples
    --------
    TODO
    --------

    References
    ----------
    .. [1] Sylvia Fruhwirth-Schnatter and Saumyadipta Pyne
           "Bayesian  Inference for Finite Mixtures of Univariate
           and Multivariate Skew Normal and Skew-t Distributions",
           Biostatistics, 11 (2010), 317-36.

    .. [2] Boris P. Hejblum, Chariff Alkhassim, Raphael Gottardo, François Caron,
           and Rodolphe Thiébaut "Sequential Dirichlet Process Mixtures of
           Multivariate Skew-t distributions for Model-based Clustering for
           model-based clustering of flow cytometry data",
           The Annals of applied statistics, Volume 13, Number 1 (2019), 638-660

    .. [3] Neal, R. M. Slice sampling. The Annals of statistics 31 (2003), 705-767

    .. [4] Walker, S.G. "Sampling the Dirichlet mixture model with slices",
               Commun. Stat., Simul. Comput. 36 (2007), 45-54

    .. [5] Kalli, M., Griffin, J. E., and Walker, S. G. "Slice sampling mixture
           models", Statistics ans Computing 21 (2011), 93-105

    .. [6] Pitman, J. "Combinatorial Stochastic Processes", volume 1875 of Lecture
           Notes in Mathematics. Springer-Verlag, Berlin Heidelberg (2006).

    .. [7] Sethuraman, J. "A constructive definition of the Dirichlet priors."
           Statistica Sinica 4 (1994), 639-650

    .. [8] West, M. "hyper parameter estimation in Dirichlet process mixture
           models", In IDSD discussion paper series (1992), 92-03.
           Duke University.

    .. [9] Abramowitz, Milton, Stegun, Irene Ann.
           "Handbook of Mathematical Functions with Formulas, Graphs,
           and Mathematical Tables". Applied Mathematics Series. 1964, 949.


    """

    def __init__(self, max_iter=100, burn_in=20, n_components_init=20,
                 max_components=100, init_params='random',
                 loc_prior=None, scaling_prior=None, scale_prior=None,
                 degrees_of_freedom_prior=None,
                 alpha_a_prior=1e-3, alpha_b_prior=1e-3, 
                 allow_singular=True,
                 random_state=None,
                 verbose=0, verbose_interval=10):
        super().__init__(
            max_iter=max_iter, burn_in=burn_in,
            n_components_init=n_components_init,
            max_components=max_components,
            init_params=init_params,
            random_state=random_state,
            verbose=verbose, verbose_interval=verbose_interval)

        self.loc_prior = loc_prior
        self.scaling_prior = scaling_prior
        self.scale_prior = scale_prior
        self.degrees_of_freedom_prior = degrees_of_freedom_prior
        self.alpha_a_prior = alpha_a_prior
        self.alpha_b_prior = alpha_b_prior

        self.allow_singular = allow_singular
 

    def _check_parameters(self, X):
        """Check that the parameters are well defined.

        Parameters
        ----------
        X : array of shape (n_observations, n_features)
        """

        self._check_loc_prior_parameter(X)
        self._check_scaling_prior_parameter(X)
        self._check_scale_prior_parameter(X)
        self._check_degrees_of_freedom_prior(X)
        self._check_alpha_priors()

    def _check_loc_prior_parameter(self, X):
        """Check the location hyper parameter of the normal-inverse-
        Whishart prior.

        Parameters
        ----------
        X : array of shape (n_observations, n_features)
        """
        _, dim = X.shape

        if self.loc_prior is None:
            self.loc_prior_ = X.mean(axis=0)
        else:
            self.loc_prior_ = check_array(self.loc_prior,
                                          dtype=[np.float64, np.float32],
                                          ensure_2d=False)
            self._check_shape(self.loc_prior_, (dim,), 'loc')

    def _check_scaling_prior_parameter(self, X):
        """Check the scaling hyper parameter of the normal-inverse-
        Whishart prior.

        Parameters
        ----------
        X : array of shape (n_observations, n_features)
        """
        _, dim = X.shape

        if self.scaling_prior is None:
            self.scaling_prior_ = dim
        else:
            _check_positive_scalar(self.scaling_prior, 'scaling_prior')
            

    def _check_degrees_of_freedom_prior(self, X):
        """Check the hyper parameter degrees of freedom of the inverse-Wishart
        prior on the scale distribution."""

        _, dim = X.shape
        if self.degrees_of_freedom_prior is None:
            self.degrees_of_freedom_prior_ = dim
        else:
            _check_positive_scalar(self.degrees_of_freedom_prior,
                                   'degrees_of_freedom_prior')
            if self.degrees_of_freedom_prior < dim:
                raise ValueError("degrees_of_freedom_prior should be, "
                                 "superior of equal to %s, but got %s, "
                                 "(dim, self.degrees_of_freedom_prior)")
            self.degrees_of_freedom_prior_ = self.degrees_of_freedom_prior
        # In 1d we shift the prior away from zero
        self.degrees_of_freedom_prior_ += 1 if dim == 1 else 0

    def _check_scale_prior_parameter(self, X):
        """Check the scale hyper parameter of the normal-inverse-
        Whishart prior.

        Parameters
        ----------
        X : array of shape (n_observations, n_features)
        """
        _, dim = X.shape

        if self.scale_prior is None:
            self.scale_prior_ = np.cov(X.T)
        else:
            self.scale_prior_ = check_array(
                self.scale_prior, dtype=[np.float64, np.float32],
                ensure_2d=False)
            self._check_shape(self.scale_prior_, (dim, dim), 'scale')
            self._check_scale_matrix(self.scale_prior)
        # used for the Wishart hierarchical prior on the scale
        if dim > 1:
            self.scale_prior_inv_ = np.linalg.inv(self.scale_prior_)
        else:
            self.scale_prior_inv_ = 1 / self.scale_prior_


    def _check_degrees_of_freedom_prior(self, X):
        """Check the hyper parameter degrees of freedom of the inverse-Wishart
        prior on the scale distribution."""

        _, dim = X.shape
        if self.degrees_of_freedom_prior is None:
            self.degrees_of_freedom_prior_ = dim
        else:
            _check_positive_scalar(self.degrees_of_freedom_prior,
                                   'degrees_of_freedom_prior')
            if self.degrees_of_freedom_prior < dim:
                raise ValueError("degrees_of_freedom_prior should be, "
                                 "superior of equal to %s, but got %s, "
                                 "(dim, self.degrees_of_freedom_prior)")
            self.degrees_of_freedom_prior_ = self.degrees_of_freedom_prior
        # In 1d we shift the prior away from zero
        self.degrees_of_freedom_prior_ += 1 if dim == 1 else 0

    def _check_alpha_priors(self):
        """Check the hyper parameters of the gamma prior on the scale factor
        (alpha) in the Dirichlet process."""

        _check_positive_scalar(self.alpha_a_prior, 'alpha_a_prior')
        _check_positive_scalar(self.alpha_b_prior, 'alpha_b_prior')
        self.alpha_a_prior_ = self.alpha_a_prior
        self.alpha_b_prior_ = self.alpha_b_prior

    def _initialize(self, X, labels):
        """Initialization of the data storage structures and
        of the mixture parameters.

        Parameters
        ----------
        X : array of shape (n_observations, n_features)

        labels : array of shape (n_observations, )
            The labels of each observation.
        """
        _, self.dim_ = X.shape

        self.labels_ = labels
        self.n_samples_ = len(labels)
        self.n_components_ = len(set(labels))
        self.n_samples_component_, _, self.ind_cl_ = \
            _clusters_sizes(labels, self.max_components)

        # gaussian parameters
        self.locs_ = np.empty(shape=(self.max_components, self.dim_),
                              dtype=float)
        self.scales_ = \
            np.empty(shape=(self.max_components, self.dim_, self.dim_),
                     dtype=float)
        

        # normal-inverse-Wishart parameters
        # locs observations of the normal component
        self.niw_locs_obs_ = np.empty(shape=(self.max_components, self.dim_),
                                       dtype=float)
        # locs of the normal component
        self.niw_locs_ = np.empty(shape=(self.max_components, self.dim_),
                                   dtype=float)
        # scalings observations of the normal component
        self.niw_scalings_obs_ = np.empty(shape=self.max_components,
                                                  dtype=float)
        # scalings of the normal component
        self.niw_scalings_ = np.empty(shape=self.max_components,
                                                  dtype=float)
        # scales observations of the inverse-Wishart component
        self.niw_scales_obs_ = np.empty(shape=(self.max_components, self.dim_,
                                                self.dim_),
                                         dtype=float)
        # scales of the inverse-Wishart component
        self.niw_scales_ = np.empty(shape=(self.max_components, self.dim_, self.dim_),
                                     dtype=float)
   
        self.niw_degrees_of_freedoms_ = np.empty(shape=self.max_components,
                                                  dtype=float)
        
        # initialize alpha
        self.alpha_ = np.log(self.n_samples_)
        self._estimate_alpha_posterior()
        # Normal-inverse-Wishart posterior parameters estimation
        self._estimate_niw_posterior_params(X)

    def _estimate_niw_posterior_params(self, X):
        """Normal-inverse-Wishart posterior
        sampling for each component.

        Parameters
        ----------
        X : array of shape (n_observations, n_features)

        References
        ----------
        .. [1] Sylvia Fruhwirth-Schnatter and Saumyadipta Pyne
           Supplementary Material to "Bayesian  Inference for Finite Mixtures
           of Univariate and Multivariate Skew Normal and Skew-t Distributions",
           Biostatistics, 11, 317-36.

        """
  
        for i, k in enumerate(self.ind_cl_):
            obs_k = np.where(self.labels_ == k)[0]
            niw_pp = \
                _niw_posterior_params(X[obs_k],
                                      self.loc_prior_,
                                      self.scaling_prior_,
                                      self.scale_prior_,
                                      self.degrees_of_freedom_prior_)
            loc, scaling, scale, degree_of_freedom = niw_pp
            # gaussian parameters
            niw_r = _niw_rvs(loc, 
                           scaling, 
                           scale,
                           degree_of_freedom, 
                           random_state=self.random_state_iter_)
            self.locs_[k], self.scales_[k] = niw_r
            loc_sample, scale_sample = niw_r
            # normal-inverse-Wishart parameters (used to evaluate the posterior)
            # locs observations of the normal component
            self.niw_locs_obs_[k] = loc_sample
            # locs of the normal component
            self.niw_locs_[k] = loc
            # scaling of the normal component
            self.niw_scalings_[k] = scaling
            # scales observations of the inverse-Wishart component
            self.niw_scales_obs_[k] = scale_sample
            # scales of the inverse-Wishart component
            self.niw_scales_[k] = scale

            self.niw_degrees_of_freedoms_[k] = degree_of_freedom

    def _stick_breaking_slice_sampler(self, X):
        """Slice sampler for the Stick breaking approach to the Dirichlet process.
        Elegantly enables a finite number of centers to be sampled within
        each iteration of the MCMC.

        Parameters
        ----------
        X : array of shape (n_observations, n_features)

        References
        ----------
        .. [1] Pitman, J. "Combinatorial Stochastic Processes", volume 1875 of Lecture
               Notes in Mathematics. Springer-Verlag, Berlin Heidelberg (2006).

        .. [2] Sethuraman, J. "A constructive definition of the Dirichlet priors."
               Statistica Sinica 4 (1994), 639-650

        .. [3] Neal, R. M. "Slice sampling", The Annals of statistics 31 (2003), 705-767

        .. [4] Walker, S.G. "Sampling the Dirichlet mixture model with slices",
               Commun. Stat., Simul. Comput. 36 (2007), 45-54

        .. [5] Kalli, M., Griffin, J. E., and Walker, S. G. "Slice sampling mixture
               models", Statistics ans Computing 21 (2011), 93-105

        """

        self.w_ = np.zeros(self.max_components, dtype=float)
        # sample the weights of each existing cluster from a Dirichlet distribution
        # and sample the rest of the weigth for potential new clusters
        # from a Gamma(alpha, 1) distribution.
        gamma_shapes = np.hstack((self.n_samples_component_[self.ind_cl_],
                                  self.alpha_ + _MACHINE_FLOAT_MIN))
        rgamma = gamma.rvs(a=gamma_shapes, size=len(gamma_shapes),
                           random_state=self.random_state_iter_)
        norm_rgamma = rgamma / rgamma.sum()
        self.w_[self.ind_cl_] = norm_rgamma[:-1]
        w_left = norm_rgamma[-1]  # weight left for potential new clusters
        # for each observation, sample a uniform random variable according to the weight
        # of its class.
        # the latent u is used in the slice sampling scheme
        self.u_ = uniform.rvs(size=self.n_samples_,
                              random_state=self.random_state_iter_) * \
                  self.w_[self.labels_]
        min_u = self.u_.min()
        # Sample the remaining weights that are needed with stick-breaking
        # i.e. the new clusters
        ind_potential_cl = np.nonzero(self.n_samples_component_ == 0)[0]  # potential new clusters
        card_new_cl = len(ind_potential_cl)

        if len(ind_potential_cl) and w_left > min_u:
            cpt_new_cl = 0  # number of new clusters
            if self.random_state_iter_ is not None:
                random_state_i = self.random_state_iter_
            else:
                random_state_i = np.random.randint(_RANDOM_STATE_UPPER)
            while w_left > min_u and cpt_new_cl < card_new_cl:
                rbeta = beta.rvs(a=1, b=self.alpha_,
                                 random_state=random_state_i)
                ind_new_cl = ind_potential_cl[cpt_new_cl]  # index of new cluster
                # weight of new cluster
                self.w_[ind_new_cl] = rbeta * w_left
                w_left = (1 - rbeta) * w_left
                random_state_i += 1
                # sample centers from prior
                loc, scale = \
                    _niw_rvs(self.loc_prior_,
                             self.scaling_prior_,
                             self.scale_prior_,
                             self.degrees_of_freedom_prior_,
                             random_state=self.random_state_iter_)
                self.locs_[ind_new_cl] = loc
                self.scales_[ind_new_cl] = scale
                cpt_new_cl += 1

        temp_ind_cl = self.w_.nonzero()[0]
        clusters_pdfs = \
            np.empty(shape=(len(temp_ind_cl), self.n_samples_), dtype=float)
        for i, k in enumerate(temp_ind_cl):
            clusters_pdfs[i, :] = \
                multivariate_normal.pdf(X, 
                                         mean=self.locs_[k],
                                         cov=self.scales_[k],
                                         allow_singular=self.allow_singular)
        # existing weights
        w_non_empty = self.w_[temp_ind_cl]
        # slices
        sliced_clusters = np.greater(w_non_empty[:, np.newaxis], self.u_)
        sliced_clusters_pdfs = clusters_pdfs * sliced_clusters
        sliced_clusters_probs = (sliced_clusters_pdfs / sliced_clusters_pdfs.
                                 sum(axis=0))
        sliced_clusters_cumuls = sliced_clusters_probs.cumsum(axis=0)
        # random_state_iter_ + 1 because variable u_ is sampled with random_state_iter_
        u_clust = uniform.rvs(size=self.n_samples_,
                              random_state=self.random_state_iter_ + 1)
        # newly sampled partition
        temp_labels = np.argmax(sliced_clusters_cumuls > u_clust, axis=0)
        # back to the original labels
        self.labels_ = temp_ind_cl[temp_labels]
        # number of observation in each cluster
        self.n_samples_component_, self.n_components_, self.ind_cl_ = \
            _clusters_sizes(self.labels_, self.max_components)
        

        # observed gaussian mixture log likelihood
        # used to compute the log posterior in _log_posterior_eval
        card_clusters = len(self.ind_cl_)
        self.clusters_pdfs_ = \
            np.zeros(shape=(card_clusters, self.n_samples_), dtype=float)

        for i, k in enumerate(self.ind_cl_):
            obs_k = np.where(self.labels_ == k)[0]
            self.clusters_pdfs_[i, obs_k] = \
                multivariate_normal.logpdf(X[obs_k],
                          mean=self.locs_[k],
                          cov=self.scales_[k],
                          allow_singular=self.allow_singular)


    def _estimate_alpha_posterior(self):
        """Posterior sampling of the Gamma prior on the scale
        parameter of the Dirichlet process.

        References
        ----------

        .. [1] West, M. "hyper parameter estimation in Dirichlet process mixture
           models", In IDSD discussion paper series (1992), 92-03.
           Duke University.

        """
        x = beta.rvs(a=self.alpha_ + 1, b=self.n_samples_,
                     random_state=self.random_state_iter_)
        pi_det = (self.alpha_a_prior_ + self.n_components_ - 1)
        pi_num = (self.n_samples_ *
                  (self.alpha_b_prior_ - np.log(x + _MACHINE_FLOAT_MIN)))
        pi = pi_det / pi_num
        pi /= 1 + pi
        if uniform.rvs(random_state=self.random_state_iter_) < pi:
            self.alpha_ = \
                gamma.rvs(a=self.alpha_a_prior_ + self.n_components_,
                          scale=1 / (self.alpha_b_prior_ -
                                     np.log(x + _MACHINE_FLOAT_MIN)),
                          random_state=self.random_state_iter_)
        else:
            self.alpha_ = \
                gamma.rvs(a=self.alpha_a_prior_ + self.n_components_ - 1,
                          scale=1 / (self.alpha_b_prior_ -
                                     np.log(x + _MACHINE_FLOAT_MIN)),
                          random_state=self.random_state_iter_)
            

    def _partition_sampling(self, X):
        """Posterior sampling of a partition.

        Parameters
        ----------
        X : array of shape (n_observations, n_features)

        """
        self._stick_breaking_slice_sampler(X)


    def _parameters_sampling(self, X):
        """Posterior sampling of a parameters set.

        Parameters
        ----------
        X : array of shape (n_observations, n_features)

        """

        self._estimate_niw_posterior_params(X)
        self._estimate_alpha_posterior()


    def _sampled_partition(self):
        """Return a sample from the dirichlet process gaussian
        mixture posterior distribution"""
        return self.labels_
    

    def _sampled_parameters(self):
        """ sampled parameters for each component of the
        gaussian mixture.

        Returns
        -------

        A dictionary containing dictionaries:
            locs : k arrays of shape (n_features, )
            shapes : k arrays of shape (n_features, )
            scales : k arrays of shape (n_features, n_features)
            degrees_of_freedoms : k floats
        """
        locs, scales = {}, {}
        for k in self.ind_cl_:
            locs[k] = self.locs_[k]
            scales[k] = self.scales_[k]

        return {'locs': locs, 'scales': scales}

    def _map_predict(self, X, map_params, map_labels):
        """Predict the labels for the data samples in X using the MAP
        of the trained model.

        Parameters
        ----------
        X : array of shape (n_observations, n_features)

        map_labels : array of shape (n_observations, )
            MAP partition.

        map_params : dict of size n_components
            MAP parameters.

        Returns
        -------
        labels : array of shape (n_observations, )
            Predicted partition.

        """
        map_locs = map_params['locs']
        map_scales = map_params['scales']

        n_observations, _ = X.shape
        n_components = len(map_locs)
        clusters_pdfs = np.zeros(shape=(n_components, n_observations), dtype=float)
        set_map_labels = np.empty(n_components, dtype=int)
        for i, (key, map_loc) in enumerate(map_locs.items()):
            clusters_pdfs[i, :] = \
                multivariate_normal.pdf(X, 
                                        mean=map_loc,
                                        cov=map_scales[key],
                                        allow_singular=self.allow_singular)
            set_map_labels[i] = key
        return set_map_labels[np.argmax(clusters_pdfs, axis=0)]
    

    def _log_posterior_eval(self):
        """Evaluate the log posterior.

        Returns
        -------

        A dictionary containing:
            mixture: float
                observed log likelihood ot the mixture
            clustering : float
                log likelihood of the clustering
            sniw_prior : float
                log likelihood of the Normal-Inverse-Wishart prior
            gamma_prior: float
                log likelihood of the gamma prior on the scale of the Dirichlet
                process.
        """
        # observed gaussian mixture log likelihood
        n_samples_component = self.n_samples_component_[self.ind_cl_]
        log_mixing = np.log(n_samples_component / self.n_samples_)
        log_lik_mixture = (self.clusters_pdfs_.sum(axis=1) +
                           n_samples_component * log_mixing).sum()
        # Chinese Restaurant Process log likelihood (clustering log likelihood)
        part0 = loggamma(self.alpha_ + _MACHINE_FLOAT_MIN)
        part1 = self.n_components_ * np.log(self.alpha_ + _MACHINE_FLOAT_MIN)
        part2 = loggamma(n_samples_component).sum()
        part3 = loggamma(self.alpha_ + self.n_samples_)
        log_lik_clustering = part0 + part1 + part2 - part3

        # normal-inverse-Wishart log likelihood
        log_lik_niw_prior = float()
        for k in self.ind_cl_:
            loc_obs = self.niw_locs_obs_[k]
            loc = self.niw_locs_[k]
            scaling = self.niw_scalings_[k]
            scale_obs = self.niw_scales_obs_[k]
            scale = self.niw_scales_[k]
            degree_of_freedom = self.niw_degrees_of_freedoms_[k]
            log_lik_niw_prior += \
                _niw_logpdf(loc_obs, 
                            scale_obs,
                            loc, 
                            scaling,
                            scale, 
                            degree_of_freedom,
                            allow_singular=self.allow_singular)
            
        # gamma prior log likelihood
        log_lik_gamma_prior = gamma.logpdf(self.alpha_ + _MACHINE_FLOAT_MIN,
                                           a=self.alpha_a_prior_,
                                           scale=self.alpha_b_prior_)

        return {'mixture': log_lik_mixture,
                'crp': log_lik_clustering,
                'niw_prior': log_lik_niw_prior,
                'gamma_prior': log_lik_gamma_prior}
    

