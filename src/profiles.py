# This module contains all the functions for generating joint distributions of susceptibility and transmissibility, as well as plotting the results.
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import multivariate_normal


def create_comm(rho, me=1.5, md=1.5, vare=1, vard=1, samplesize=None):
    """
    Docstring for create_comm

    :param rho: Description
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
    ax = sns.heatmap(joint_dist, xticklabels=9, yticklabels=9)
    ax.set(xlabel="susceptibility", ylabel="transmisibility")
    ax.set_xticklabels(np.round([x[i] for i in range(0, 100, 9)], 2), rotation=90)
    ax.set_yticklabels(np.round([y[i] for i in range(0, 100, 9)], 2), rotation=0)
    ax.xaxis.tick_top()
    plt.show()
