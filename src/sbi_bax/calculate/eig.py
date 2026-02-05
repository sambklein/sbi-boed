import logging
import os
import numpy as np
import torch
from typing import Callable, Optional
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from contextlib import redirect_stderr, redirect_stdout

from sbi_bax.calculate.entropy import histogram_entropy

log = logging.getLogger(__name__)


def check_results(results):
    """
    Check if any of the results contain NaN or infinite values.
    If so, raise an error with the problematic results.
    """
    eigs = np.array([r[0] for r in results])
    if any(np.isnan(eigs)) or any(np.isinf(eigs)):
        log.warning("NaN or infinite values found in EIG calculations.")
        raise ValueError("EIG calculation resulted in NaN or infinite values.")
    if any(eigs < 0):
        log.warning(
            "Negative EIG values found, which may indicate issues with the model or data."
        )
        # Report the smallest EIG value.
        min_eig = min(eigs)
        log.warning(f"Smallest EIG value: {min_eig:.4f}")


def eig_weights(
    x_all: torch.Tensor,
    simulator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ea_paths: torch.Tensor,
    data_posterior: NeuralPosterior,
    ea_posterior: NeuralPosterior,
    n_mc_theta: int = 1000,
    entropy_estimator: Callable[
        [np.ndarray, Optional[np.ndarray]], float
    ] = histogram_entropy,
    return_entropies: bool = False,
) -> float:
    """
    Calculate the expected information gain (EIG) for a given input x.

    Computes EIG = H(y_x | D_t) - E_{e_A}[H(y_x | D_t, e_A)] using Monte Carlo estimation.
    This measures how much information we expect to gain about the parameter θ by
    observing y_x at input x, given the current data D_t and execution paths e_A.

    Parameters
    ----------
    x : torch.Tensor
        Input location where we want to calculate the expected information gain.
    simulator : Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]
        Function that simulates observations y given input x and parameter θ.
        Should accept (x, theta_samples) and return simulated y values.
    ea_paths : torch.Tensor
        Collection of execution paths e_A used for conditioning the posterior.
    data_posterior : Distribution
        Posterior distribution p(θ | D_t) given current data D_t.
        Should have a .sample(n_samples) method that returns θ samples.
    data_ea_posterior : NeuralPosterior
        Posterior distribution p(θ | D_t, e_A) given current data D_t and execution paths e_A.
    n_mc_sim : int, optional
        Number of Monte Carlo simulations for y_x sampling, by default 1000.
        Currently unused but reserved for future implementation.
    n_mc_theta : int, optional
        Number of Monte Carlo samples from p(θ | D_t), by default 1000.
    entropy_estimator : Callable[[np.ndarray, Optional[np.ndarray]], float], optional
        Function to estimate entropy from samples. Should accept samples and optional
        weights for weighted entropy estimation. Default is histogram_entropy.
    return_entropies : bool, optional
        If True, return tuple (eig, H(y_x|D_t), E[H(y_x|D_t,e_A)]).
        If False, return only the EIG value. Default is False.

    Returns
    -------
    float or tuple
        If return_entropies=False: Expected information gain (EIG) at x.
        If return_entropies=True: Tuple of (EIG, marginal_entropy, conditional_entropy).

    Notes
    -----
    The EIG is computed as the difference between:
    1. H(y_x | D_t): Entropy of predictions at x given current data
    2. E_{e_A}[H(y_x | D_t, e_A)]: Expected entropy after conditioning on execution paths

    Higher EIG values indicate that observing at x would provide more information
    for distinguishing between different experimental designs in ea_paths.
    """
    # Generate samples from p(y_x | D_t) = \int d\theta p(y_x | \theta) p(\theta | D_t).
    with torch.no_grad():
        with open(os.devnull, "w") as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                dt_thetas = data_posterior.sample((n_mc_theta,)).cpu()
                # Generate samples from p(y_x | D_t, e_A)
                dt_ea_thetas = []
                for ea_path in ea_paths:
                    ea_posterior.set_default_x(ea_path)
                    dt_ea_thetas.append(ea_posterior.sample((n_mc_theta,)))
                # Concatenate thetas from all execution paths.
                dt_ea_thetas = torch.stack(dt_ea_thetas, dim=0).view(
                    -1, dt_thetas.size(-1)
                )
                # Randomly sample n_mc_theta thetas from the execution paths.
                dt_ea_thetas = dt_ea_thetas[
                    torch.randint(0, dt_ea_thetas.size(0), (n_mc_theta,))
                ]
                # Get the likelihoods for these execution paths.
                dt_ea_likelihoods = []
                for ea_path in ea_paths:
                    ea_posterior.set_default_x(ea_path)
                    dt_ea_likelihoods.append(
                        ea_posterior.log_prob(dt_ea_thetas, norm_posterior=False)
                        .exp()
                        .cpu()
                    )
                # Get the average likelihood across all execution paths.
                dt_ea_likelihood = torch.stack(dt_ea_likelihoods, dim=0).mean(dim=0)
                # Divide by the likelihoods of each path to get the weights.
                weights = torch.stack(dt_ea_likelihoods, dim=0) / dt_ea_likelihood
                # Move the dt_ea_thetas to the CPU for entropy estimation.
                dt_ea_thetas = dt_ea_thetas.cpu()

        # Print the average effective sample size for each path.
        log.info("Weight statistics:")
        aw = []
        for i, w in enumerate(weights):
            aw.append((w.sum() ** 2 / (w**2).sum()).item())
        log.info(f"Average effective sample size: {np.mean(aw):.1f} ± {np.std(aw):.1f}")

        results = []
        for x_eval in x_all:
            yx_samples = simulator(x_eval.view(1, -1), dt_thetas).cpu().numpy()
            # Get an MC estimate for H(y_x | D_t).
            h_yx_dt = entropy_estimator(yx_samples)
            # Estimate the second term in the EIG, which is the expected entropy of p(y_x | D_t, e_A).
            h_ea_dt = []
            # Generate samples using dt_ea_thetas and the weights.
            yx_samples = simulator(x_eval.view(1, -1), dt_ea_thetas).cpu().numpy()
            # Calculate the entropy of p(y_x | D_t, e_A) using these samples.
            for weight in weights:
                # Calculate the entropy of p(y_x | D_t, e_A) using these weights.
                h_ea_dt += [entropy_estimator(yx_samples, weights=weight)]

            # This is an MC estimate for the second term in the EIG.
            e_h_ea = np.mean(h_ea_dt)
            # Calculate the expected information gain (EIG).
            eig_x = h_yx_dt - e_h_ea
            results.append([eig_x, h_yx_dt, e_h_ea])
        # Check if any of the eigs are NaN or infinite.
        check_results(results)

        if return_entropies:
            # Return the entropies used to calculate the EIG.
            return zip(*results)
        else:
            # Return only the EIG.
            return torch.tensor(list(zip(*results))[0])


def eig_direct(
    x_all: torch.Tensor,
    simulator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ea_paths: torch.Tensor,
    data_posterior: NeuralPosterior,
    ea_posterior: NeuralPosterior,
    n_mc_theta: int = 1000,
    entropy_estimator: Callable[
        [np.ndarray, Optional[np.ndarray]], float
    ] = histogram_entropy,
    return_entropies: bool = False,
) -> float:
    """
    Calculate the expected information gain (EIG) for a given input x.

    Computes EIG = H(y_x | D_t) - E_{e_A}[H(y_x | D_t, e_A)] using Monte Carlo estimation.
    This measures how much information we expect to gain about the parameter θ by
    observing y_x at input x, given the current data D_t and execution paths e_A.

    Parameters
    ----------
    x : torch.Tensor
        Input location where we want to calculate the expected information gain.
    simulator : Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]
        Function that simulates observations y given input x and parameter θ.
        Should accept (x, theta_samples) and return simulated y values.
    ea_paths : torch.Tensor
        Collection of execution paths e_A used for conditioning the posterior.
    data_posterior : Distribution
        Posterior distribution p(θ | D_t) given current data D_t.
        Should have a .sample(n_samples) method that returns θ samples.
    ea_posterior : NeuralPosterior
        Posterior distribution p(θ | D_t, e_A) given current data D_t and execution paths e_A.
        Should have a .sample(n_samples) method that returns θ samples.
    n_mc_sim : int, optional
        Number of Monte Carlo simulations for y_x sampling, by default 1000.
        Currently unused but reserved for future implementation.
    n_mc_theta : int, optional
        Number of Monte Carlo samples from p(θ | D_t), by default 1000.
    entropy_estimator : Callable[[np.ndarray, Optional[np.ndarray]], float], optional
        Function to estimate entropy from samples. Should accept samples and optional
        weights for weighted entropy estimation. Default is histogram_entropy.
    return_entropies : bool, optional
        If True, return tuple (eig, H(y_x|D_t), E[H(y_x|D_t,e_A)]).
        If False, return only the EIG value. Default is False.

    Returns
    -------
    float or tuple
        If return_entropies=False: Expected information gain (EIG) at x.
        If return_entropies=True: Tuple of (EIG, marginal_entropy, conditional_entropy).

    Notes
    -----
    The EIG is computed as the difference between:
    1. H(y_x | D_t): Entropy of predictions at x given current data
    2. E_{e_A}[H(y_x | D_t, e_A)]: Expected entropy after conditioning on execution paths

    Higher EIG values indicate that observing at x would provide more information
    for distinguishing between different experimental designs in ea_paths.
    """
    # Generate samples from p(y_x | D_t) = \int d\theta p(y_x | \theta) p(\theta | D_t).
    with torch.no_grad():
        with open(os.devnull, "w") as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                dt_thetas = data_posterior.sample((n_mc_theta,)).cpu()
                dt_ea_thetas = []
                for ea_path in ea_paths:
                    ea_posterior.set_default_x(ea_path)
                    dt_ea_thetas.append(ea_posterior.sample((n_mc_theta,)).cpu())

        results = []
        for x_eval in x_all:
            yx_samples = simulator(x_eval.view(1, -1), dt_thetas).cpu().numpy()
            # Get an MC estimate for H(y_x | D_t).
            h_yx_dt = entropy_estimator(yx_samples)
            # Estimate the second term in the EIG, which is the expected entropy of p(y_x | D_t, e_A).
            h_ea_dt = []
            for dt_ea_theta in dt_ea_thetas:
                yx_samples = simulator(x_eval.view(1, -1), dt_ea_theta).cpu().numpy()
                # Calculate the entropy of p(y_x | D_t, e_A) using these samples.
                h_ea_dt += [entropy_estimator(yx_samples)]

            # This is an MC estimate for the second term in the EIG.
            e_h_ea = np.mean(h_ea_dt)
            # Calculate the expected information gain (EIG).
            eig_x = h_yx_dt - e_h_ea
            results.append([eig_x, h_yx_dt, e_h_ea])
        # Check if any of the eigs are NaN or infinite.
        check_results(results)
        if return_entropies:
            # Return the entropies used to calculate the EIG.
            return zip(*results)
        else:
            # Return only the EIG.
            return torch.tensor(list(zip(*results))[0])
