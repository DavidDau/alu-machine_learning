#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def variance(X, C):
    """[summary]

    Args:
        X ([type]): [description]
        C ([type]): [description]

    Returns:
        [type]: [description]
    """
    try:
        k, d = C.shape
        aux1 = np.sum(C ** 2, axis=1)[:, np.newaxis]
        aux2 = np.sum(X ** 2, axis=1)
        aux3 = np.matmul(C, X.T)
        SED = aux1 - 2 * aux3 + aux2
        return np.sum(np.amin(SED, axis=0))
    except Exception:
        return None
