#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """[summary]

    Args:
        Observation ([type]): [description]
        Emission ([type]): [description]
        Transition ([type]): [description]
        Initial ([type]): [description]

    Returns:
        [type]: [description]
    """
    if type(Observation) is not np.ndarray or Observation.ndim != 1:
        return None, None
    if type(Emission) is not np.ndarray or Emission.ndim != 2:
        return None, None
    N, M = Emission.shape
    if (type(Transition) is not np.ndarray or Transition.ndim != 2 or
        Transition.shape[0] != Transition.shape[1] or
            Transition.shape[0] != N):
        return None, None
    if (type(Initial) is not np.ndarray or Initial.ndim != 2 or
            Initial.shape != (N, 1)):
        return None, None
    B = np.zeros((N, Observation.shape[0]))
    B[:, Observation.shape[0] - 1] = 1
    for k in range(Observation.shape[0] - 2, -1, -1):
        for j in range(N):
            B[j, k] = 0
            for l in range(N):
                B[j, k] += (B[l, k + 1] * Transition[j, l] *
                            Emission[l, Observation[k + 1]])
    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])
    return P, B
