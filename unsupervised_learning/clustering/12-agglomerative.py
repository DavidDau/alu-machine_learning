#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """[summary]

    Args:
        X ([type]): [description]
        dist ([type]): [description]

    Returns:
        [type]: [description]
    """
    linked = scipy.cluster.hierarchy.linkage(X, 'ward')
    clss = scipy.cluster.hierarchy.fcluster(linked, t=dist,
                                            criterion='distance')
    plt.figure()
    scipy.cluster.hierarchy.dendrogram(linked, color_threshold=dist)
    plt.show()
    return clss
