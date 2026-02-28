#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def absorbing(P):
    """[summary]

    Args:
        P ([type]): [description]

    Returns:
        [type]: [description]
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    s1, s2 = P.shape
    if s1 != s2:
        return None
    if np.sum(P, axis=1).all() != 1:
        return None
    if (np.diag(P) == 1).all():
        return True
    if not (np.diag(P) == 1).any():
        return False
    for i in range(s1):
        if P[i][i] == 1:
            return True
    return False
