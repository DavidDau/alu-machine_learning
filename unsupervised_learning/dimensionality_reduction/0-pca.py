#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def pca(X, var=0.95):
    """[summary]

    Args:
        X ([type]): [description]
        var (float, optional): [description]. Defaults to 0.95.

    Returns:
        [type]: [description]
    """
    U, S, Vh = np.linalg.svd(X)

    cum = np.cumsum(S)

    idx = len(cum) - 1
    for i in range(len(cum)):
        if cum[i] / cum[-1] >= var:
            idx = i
            break

    return Vh.T[:, :i + 1]
