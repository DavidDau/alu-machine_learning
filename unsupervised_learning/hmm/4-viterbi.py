#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """[summary]

    Args:
        Observation ([type]): [description]
        Emission ([type]): [description]
        Transition ([type]): [description]
        Initial ([type]): [description]

    Returns:
        [type]: [description]
    """
    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    T = Observation.shape[0]
    N, M = Emission.shape
    if Transition.shape[0] != N or Transition.shape[1] != N:
        return None, None
    if Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None
    if not np.sum(Emission, axis=1).all():
        return None, None
    if not np.sum(Transition, axis=1).all():
        return None, None
    if not np.sum(Initial) == 1:
        return None, None
    viterbi = np.zeros((N, T))
    aux = np.zeros((N, T))
    Obs_t = Observation[0]
    aux[:, 0] = 0
    x1 = np.multiply(Initial[:, 0], Emission[:, Obs_t])
    viterbi[:, 0] = x1
    for t in range(1, T):
        a = viterbi[:, t - 1]
        b = Transition.T
        ab = a * b
        ab_max = np.amax(ab, axis=1)
        c = Emission[:, Observation[t]]
        x1 = ab_max * c

        viterbi[:, t] = x1
        aux[:, t - 1] = np.argmax(ab, axis=1)
    aux2 = []
    c_A = np.argmax(viterbi[:, T - 1])
    aux2 = [c_A] + aux2
    for t in range(T - 2, -1, -1):
        c_A = int(aux[c_A, t])
        aux2 = [c_A] + aux2
    P = np.amax(viterbi[:, T - 1], axis=0)
    return aux2, P
