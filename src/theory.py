# This module contains all of the theoretical calculations for the model, including the basic reproduction number and the infection rate.
import numpy as np


def BC_r0(a, b, beta, gamma, k, s):
    """Function to calculate R0 for a population with two
    balanced communities, given the parameters of the model.

    Parameters
    ----------
    a : float
        The weighted covariance of susceptibility and
        transmissibility for the first community.
    b : float
        The weighted covariance of susceptibility and
        transmissibility for the second community.
    beta : float
        The transmission rate.
    gamma : float
        The recovery rate.
    k : float
        The average degree of the network.
    s : float
        The imbalance parameter.

    Returns
    -------
    float
        The basic reproduction number.
    """
    err = ((1 + s) * (a - b)) ** 2 + 4 * a * b * (1 - s) ** 2
    l = (1 + s) * (a + b) / 4 + np.sqrt(max(err, 0)) / 4
    return beta * k * l / gamma


def BC_beta(a, b, R0, gamma, k, s):
    """Function to calculate the transmission rate for a population
    with two balanced communities, given the parameters of the model.

    Parameters
    ----------
    a : float
        The weighted covariance of susceptibility
        and transmissibility for the first community.
    b : float
        The weighted covariance of susceptibility
        and transmissibility for the second community.
    R0 : float
        The basic reproduction number.
    gamma : float
        The recovery rate.
    k : float
        The average degree of the network.
    s : float
        The imbalance parameter.

    Returns
    -------
    float
        The transmission rate.
    """
    err = ((1 + s) * (a - b)) ** 2 + 4 * a * b * (1 - s) ** 2
    l = (1 + s) * (a + b) / 4 + np.sqrt(max(err, 0)) / 4
    return (R0 * gamma) / (k * l)


def IBC_r0(a, b, beta, gamma, k, s, r):
    """Function to calculate R0 for a population with
    two imbalanced communities, given the parameters of the model.

    Parameters
    ----------
    a : float
        The weighted covariance of susceptibility
        and transmissibility for the first community.
    b : float
        The weighted covariance of susceptibility
        and transmissibility for the second community.
    beta : float
        The transmission rate.
    gamma : float
        The recovery rate.
    k : float
        The average degree of the network.
    s : float
        The imbalance parameter.
    r : float
        The proportion of individuals in the first community.

    Returns
    -------
    float
        The basic reproduction number.
    """
    alpha = (1 - (r**2 + (1 - r) ** 2)) / (r**2 + (1 - r) ** 2)
    err = ((1 + alpha * s) * (r * a - (1 - r) * b)) ** 2 + (
        4 * r * (1 - r) * a * b * (1 - s) ** 2
    )
    l = ((1 + alpha * s) * (r * a + (1 - r) * b)) / 2 + np.sqrt(max(err, 0)) / 2
    return beta * k * l / gamma


def IBC_beta(a, b, R0, gamma, k, s, r):
    """Function to calculate the transmission rate
    for a population with two imbalanced comunities.

    Parameters
    ----------
    a : float
        The weighted covariance of susceptibility
        and transmissibility for the first community.
    b : float
        The weighted covariance of susceptibility
        and transmissibility for the second community.
    R0 : float
        The basic reproduction number.
    gamma : float
        The recovery rate.
    k : float
        The average degree of the network.
    s : float
        The imbalance parameter.
    r : float
        The proportion of individuals in the first community.

    Returns
    -------
    float
        The transmission rate.
    """
    x = k * (1 + ((1 - r**2 - (1 - r) ** 2) / (r**2 + (1 - r) ** 2)) * s)
    y = k * (1 - s)
    err = (r * x * a + (1 - r) * x * b) ** 2 - 4 * (r * (1 - r) * a * b * (x**2 - y**2))
    l = (r * x * a + (1 - r) * x * b) / 2 + np.sqrt(max(err, 0)) / 2
    return (R0 * gamma) / l
