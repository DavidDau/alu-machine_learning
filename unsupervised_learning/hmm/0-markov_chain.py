#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
    """
import numpy as np


def markov_chain(P, s, t=1):
    """[summary]

    Args:
        P ([type]): [description]
        s ([type]): [description]
        t (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if np.sum(P, axis=1).all() != 1:
        return None
    if not isinstance(t, int) or t < 0:
        return None
    if not isinstance(s, np.ndarray) or len(s.shape) != 2:
        return None
    if s.shape[0] != 1 or s.shape[1] != P.shape[0]:
        return None
    if t == 0:
        return s
    no_d = np.matmul(s, P)
    for i in range(1, t):
        no_d = np.matmul(no_d, P)
    return no_d
