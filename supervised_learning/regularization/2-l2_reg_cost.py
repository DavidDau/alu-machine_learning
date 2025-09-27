#!/usr/bin/env python3
"""
Function calculates the L2 Regularization Cost
"""


import tensorflow as tf


def l2_reg_cost(cost):
    """
    Calculates the cost of a neural network with L2 regularization

    Args:
        cost (float): is a tensor containing the cost of the network
        without L2 regularization

    Returns:
    a tensor containing the cost
    """
    return cost + tf.losses.get_regularization_losses()
