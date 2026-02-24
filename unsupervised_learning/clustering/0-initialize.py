#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""

import numpy as np


def initialize(X, k):
    """[summary]

    Args:
        X ([type]): [description]
        k ([type]): [description]

    Returns:
        [type]: [description]
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if type(k) != int or k <= 0 or k >= X.shape[0]:
        return None

    n, d = X.shape

    min_ = np.amin(X, axis=0)
    max_ = np.amax(X, axis=0)

    return np.random.uniform(min_, max_, (k, d))
