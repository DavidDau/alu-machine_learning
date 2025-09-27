#!/usr/bin/env python3
"""
Function creates a Tensorflow Layer with L2 Regularization
"""


import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a Layer with L2 Regularization

    Args:
        prev (_type_): _description_
        n (_type_): _description_
        activation (_type_): _description_
        lambtha (_type_): _description_

    Returns:
    new layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    reg = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=reg)
    return layer(prev)
