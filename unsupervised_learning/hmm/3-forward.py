#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
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
    T = Observation.shape[0]
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
    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    for i in range(1, T):
        state = F[:, i - 1].T @ Transition
        F[:, i] = state * Emission[:, Observation[i]]
    P = np.sum(F[:, -1])
    return P, F
