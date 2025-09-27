#!/usr/bin/env python3
"""
Function calculates the cost of a neural network with L2 regularization
"""


import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization

    Args:
    - cost: cost of the network without l2 regularization
    - lambtha: regularization parameter
    - weights: dictionary of weights and biases
    - L: number of layers
    - m: number of data points used

    Returns:
    cost of the network
    """
    l2_term = 0

    # Sum the squared norms of the weights
    for i in range(1, L + 1):
        weight_key = 'W' + str(i)
        if weight_key in weights:
            l2_term += np.sum(np.square(weights[weight_key]))

    # Compute the L2 regularization term
    l2_term *= (lambtha / (2 * m))

    # Add the L2 regularization term to the original cost
    l2_cost = cost + l2_term

    return l2_cost
