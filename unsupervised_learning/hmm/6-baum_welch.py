#!/usr/bin/env python3
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """[summary]

    Args:
        Observations ([type]): [description]
        Transition ([type]): [description]
        Emission ([type]): [description]
        Initial ([type]): [description]
        iterations (int, optional): [description]. Defaults to 1000.

    Returns:
        [type]: [description]
    """
    if not isinstance(Observations, np.ndarray) \
            or len(Observations.shape) != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None
    T = Observations.shape[0]
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
    for n in range(iterations):
        _, alpha = forward(Observations, Emission, Transition, Initial)
        _, B__ = backward(Observations, Emission, Transition, Initial)
        xi = np.zeros((N, N, T - 1))
        for t in range(T - 1):
            a = np.matmul(alpha[:, t].T, Transition)
            b = Emission[:, Observations[t + 1]].T
            c = B__[:, t + 1]
            denominator = np.matmul(a * b, c)
            for i in range(N):
                a = alpha[i, t]
                b = Transition[i]
                c = Emission[:, Observations[t + 1]].T
                d = B__[:, t + 1].T
                numerator = a * b * c * d
                xi[i, :, t] = numerator / denominator
        x_g = np.sum(xi, axis=1)
        num = np.sum(xi, 2)
        den = np.sum(x_g, axis=1).reshape((-1, 1))
        Transition = num / den
        sum_i = np.sum(xi[:, :, T - 2], axis=0)
        sum_i = sum_i.reshape((-1, 1))
        x_g = np.hstack((x_g, sum_i))
        denominator = np.sum(x_g, axis=1)
        denominator = denominator.reshape((-1, 1))
        for i in range(M):
            x_g_i = x_g[:, Observations == i]
            Emission[:, i] = np.sum(x_g_i, axis=1)
        Emission = Emission / denominator
    return Transition, Emission


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
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
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
    B__ = np.zeros((N, T))
    B__[:, T - 1] = np.ones(N)
    for t in range(T - 2, -1, -1):
        a = Transition
        b = Emission[:, Observation[t + 1]]
        c = B__[:, t + 1]
        abc = a * b * c
        prob = np.sum(abc, axis=1)
        B__[:, t] = prob
    P_first = Initial[:, 0] * Emission[:, Observation[0]] * B__[:, 0]
    P = np.sum(P_first)
    return P, B__


def forward(Observation, Emission, Transition, Initial):
    """[summary]

    Args:
        Observations ([type]): [description]
        Transition ([type]): [description]
        Emission ([type]): [description]
        Initial ([type]): [description]
        iterations (int, optional): [description]. Defaults to 1000.

    Returns:
        [type]: [description]
    """
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
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
    F = np.zeros((N, T))
    Obs_i = Observation[0]
    prob = np.multiply(Initial[:, 0], Emission[:, Obs_i])
    F[:, 0] = prob
    for i in range(1, T):
        Obs_i = Observation[i]
        state = np.matmul(F[:, i - 1], Transition)
        prob = np.multiply(state, Emission[:, Obs_i])
        F[:, i] = prob
    return np.sum(F[:, T - 1]), F
