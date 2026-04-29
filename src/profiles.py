# This module contains all the functions for generating joint distributions of susceptibility and transmissibility, as well as plotting the results.
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import multivariate_normal


def create_comm(rho, me=1.5, md=1.5, vare=1, vard=1, samplesize=None):
    """Function used to create the joint distribution of susceptibility
    and transmissibility, given a correlation coefficient and other parameters.
    The distribution is normalized to sum to 1, and can also return samples from
    the distribution if a sample size is provided.

    Parameters
    ----------
    rho : float
        The correlation coefficient between susceptibility and transmissibility. Must be between -1 and 1.
    me : float, optional
        The mean of the susceptibility distribution, by default 1.5
    md : float, optional
        The mean of the transmissibility distribution, by default 1.5
    vare : int, optional
        The variance of the susceptibility distribution, by default 1
    vard : int, optional
        The variance of the transmissibility distribution, by default 1
    samplesize : int, optional
        The number of samples to draw from the distribution, by default None

    Returns
    -------
    np.ndarray
        The joint distribution of susceptibility and transmissibility, normalized to sum to 1.
    np.ndarray, optional
        If samplesize is provided, returns a tuple of the joint distribution and the samples drawn from the distribution.
    """
    x = np.linspace(0.1, 3, 100)
    y = np.linspace(0.1, 3, 100)

    X, Y = np.meshgrid(x, y)

    mean_x = me
    mean_y = md

    std_x = vare**0.5
    std_y = vard**0.5

    cov_xy = rho * std_x * std_y

    # 3D matrix with each slice being X and Y, so all the combos
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # Defining the mean and cov matrix for our multivariate
    mean = [mean_x, mean_y]
    covariance_matrix = [[std_x**2, cov_xy], [cov_xy, std_y**2]]

    rv = multivariate_normal(mean=mean, cov=covariance_matrix, allow_singular=True)

    joint_dist = rv.pdf(pos)
    if samplesize:
        samples = rv.rvs(size=samplesize * 5)
        in_bounds = np.all((samples >= x[0]) & (samples <= x[-1]), axis=1)
        final_samples = samples[in_bounds]
        return joint_dist / np.sum(joint_dist), final_samples[:samplesize]
    return joint_dist / np.sum(joint_dist)


def plot_comm(joint_dist, x, y):
    """Function to plot a joint distribution.

    Parameters
    ----------
    joint_dist : np.ndarray
         The joint distribution to be plotted, as a 2D array.
    x : np.ndarray
        The x-axis values for the plot.
    y : np.ndarray
        The y-axis values for the plot.
    """
    ax = sns.heatmap(joint_dist, xticklabels=9, yticklabels=9)
    ax.set(xlabel="susceptibility", ylabel="transmisibility")
    ax.set_xticklabels(np.round([x[i] for i in range(0, 100, 9)], 2), rotation=90)
    ax.set_yticklabels(np.round([y[i] for i in range(0, 100, 9)], 2), rotation=0)
    ax.xaxis.tick_top()
    plt.show()
