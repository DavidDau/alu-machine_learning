#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def pca(X, ndim):
    """[summary]

    Args:
        X ([type]): [description]
        ndim ([type]): [description]

    Returns:
        [type]: [description]
    """
    _, _, v = np.linalg.svd(X - np.mean(X, axis=0))
    return np.matmul(X - np.mean(X, axis=0), v[:ndim].T)
