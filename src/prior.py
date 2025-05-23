# -----------------------------
# prior.py
# -----------------------------

import numpy as np


def exponential_model_prior(max_degree, lam=0.1):
    """
    Returns an exponential prior over model degrees 1, ..., max_degree.
    P(m) ~ exp(-lambda * m)
    """
    degrees = np.arange(1, max_degree + 1)
    prior = np.exp(-lam * degrees)
    return prior / np.sum(prior)


def uniform_model_prior(max_degree):
    """
    Returns a uniform prior over model degrees 1, ..., max_degree.
    """
    return np.ones(max_degree) / max_degree


def posterior_over_models(chi2_values, prior, sigma_squared):
    """
    Computes the posterior probability of each model degree given chi2 values and a prior.

    Posterior P(m|D) ~ exp(-chi2(m) / 2sigma^2) * Prior(m)
    """
    log_likelihood = -0.5 * chi2_values / sigma_squared
    log_posterior_unnorm = log_likelihood + np.log(prior)

    # Normalize with log-sum-exp trick
    log_posterior_shifted = log_posterior_unnorm - np.max(log_posterior_unnorm)
    posterior_unnorm = np.exp(log_posterior_shifted)
    posterior = posterior_unnorm / np.sum(posterior_unnorm)
    return posterior