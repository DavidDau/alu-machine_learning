#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """[summary]

    Args:
        X ([type]): [description]
        k ([type]): [description]
        iterations (int, optional): [description]. Defaults to 1000.

    Returns:
        [type]: [description]
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if type(k) != int or k <= 0:
        return None, None

    if type(iterations) != int or iterations <= 0:
        return None, None
    n, d = X.shape
    minimum = np.amin(X, axis=0)
    maximum = np.amax(X, axis=0)
    C = initialize(X, k)
    clss = None
    for i in range(iterations):
        C_cpy = np.copy(C)
        distance = np.linalg.norm(X[:, None] - C, axis=-1)
        clss = np.argmin(distance, axis=-1)
        # move the centroids
        for j in range(k):
            index = np.argwhere(clss == j)
            if not len(index):
                C[j] = initialize(X, 1)
            else:
                C[j] = np.mean(X[index], axis=0)
        if (C_cpy == C).all():
            return C, clss
    distance = np.linalg.norm(X[:, None] - C, axis=-1)
    clss = np.argmin(distance, axis=-1)

    return C, clss


def initialize(X, k):
    """[summary]

    Args:
        X ([type]): [description]
        k ([type]): [description]

    Returns:
        [type]: [description]
    """
    n, d = X.shape
    minimum = np.amin(X, axis=0)
    maximum = np.amax(X, axis=0)
    values = np.random.uniform(minimum, maximum, (k, d))
    return values
