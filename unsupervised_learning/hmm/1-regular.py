#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def regular(P):
    """[summary]

    Args:
        P ([type]): [description]

    Returns:
        [type]: [description]
    """
    if (type(P) is not np.ndarray or P.ndim != 2 or
            P.shape[0] != P.shape[1]):
        return None
    n = P.shape[0]
    if (P < 0).any() or (P > 1).any():
        return None
    aux1, evects = np.linalg.eig(P.T)
    x_e = np.argmin(np.abs(aux1 - 1))
    if not np.isclose(aux1[x_e],  1):
        return None
    if np.sum(np.isclose(aux1, 1)) != 1:
        return None
    res = evects[:, x_e].reshape(1, -1)
    return res / np.sum(res)
