# This module contains all of the theoretical calculations for the model, including the basic reproduction number and the infection rate.
import numpy as np


def BC_r0(a, b, beta, gamma, k, e):
    err = ((1 + e) * (a - b)) ** 2 + 4 * a * b * (1 - e) ** 2
    l = (1 + e) * (a + b) / 4 + np.sqrt(max(err, 0)) / 4
    return beta * k * l / gamma


def BC_beta(a, b, R0, gamma, k, e):
    err = ((1 + e) * (a - b)) ** 2 + 4 * a * b * (1 - e) ** 2
    l = (1 + e) * (a + b) / 4 + np.sqrt(max(err, 0)) / 4
    return (R0 * gamma) / (k * l)


def IBC_r0(a, b, beta, gamma, k, e, r):
    alpha = (1 - (r**2 + (1 - r) ** 2)) / (r**2 + (1 - r) ** 2)
    err = ((1 + alpha * e) * (r * a - (1 - r) * b)) ** 2 + (
        4 * r * (1 - r) * a * b * (1 - e) ** 2
    )
    l = ((1 + alpha * e) * (r * a + (1 - r) * b)) / 2 + np.sqrt(max(err, 0)) / 2
    return beta * k * l / gamma


def IBC_beta(a, b, R0, gamma, k, e, r):
    x = k * (1 + ((1 - r**2 - (1 - r) ** 2) / (r**2 + (1 - r) ** 2)) * e)
    y = k * (1 - e)
    # err = (c11 * (a + b))**2 - 4*a*b*(c11**2 - c12**2)
    # l = c11 * (a + b) / 2 + np.sqrt(max(err, 0)) / 2
    err = (r * x * a + (1 - r) * x * b) ** 2 - 4 * (r * (1 - r) * a * b * (x**2 - y**2))
    # l = (r*x*a + (1-r)*x*b) / 2 + np.sqrt(max(err, 0)) / 2
    l = (r * x * a + (1 - r) * x * b) / 2 + np.sqrt(max(err, 0)) / 2
    return (R0 * gamma) / l


def g0(a, b, beta, r0, k, e):
    c11 = (beta * k * (1 + e)) / 2
    c12 = (beta * k * (1 - e)) / 2
    # err = (c11 * (a + b))**2 - 4*a*b*(c11**2 - c12**2)
    # l = c11 * (a + b) / 2 + np.sqrt(max(err, 0)) / 2
    err = (c11 * (a - b)) ** 2 + 4 * a * b * c12**2
    l = c11 * (a + b) / 2 + np.sqrt(max(err, 0)) / 2
    return l / r0
