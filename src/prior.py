# -----------------------------
# src/prior.py
# -----------------------------
import numpy as np
from scipy.special import logsumexp

def exponential_model_prior(max_degree, lam=0.3):
    """
    Returns a prior probability distribution over model degrees 1 to max_degree.
    Prior: P(m) ~ exp(-lambda * m)

    Parameters:
        max_degree (int): Maximum polynomial degree (inclusive).
        lam (float): Decay rate of the exponential prior.

    Returns:
        np.ndarray: Prior probabilities of shape (max_degree,)
    """
    degrees = np.arange(1, max_degree + 1)
    unnormalized = np.exp(-lam * degrees)
    return unnormalized / np.sum(unnormalized)

def uniform_model_prior(max_degree):
    """
    Returns a uniform prior distribution over model degrees 1 to max_degree.

    Parameters:
        max_degree (int): Maximum polynomial degree (inclusive).

    Returns:
        np.ndarray: Uniform prior probabilities of shape (max_degree,)
    """
    return np.ones(max_degree) / max_degree

def posterior_over_models(chi2_values, prior, sigma_squared=1.0):
    """
    Computes the posterior distribution over model degrees given chi-squared values and a prior.

    P(m|D) ∝ exp(-0.5 * chi2 / sigma^2) * P(m)

    Parameters:
        chi2_values (array-like): Chi-squared values for each model.
        prior (array-like): Prior probability for each model.
        sigma_squared (float): Variance of the noise.

    Returns:
        np.ndarray: Normalized posterior distribution over models.
    """
    log_likelihood = -0.5 * np.array(chi2_values) / sigma_squared
    log_posterior_unnormalized = log_likelihood + np.log(prior)
    log_posterior_normalized = log_posterior_unnormalized - logsumexp(log_posterior_unnormalized)
    posterior = np.exp(log_posterior_normalized)
    return posterior
