#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import sklearn.cluster


def kmeans(X, k):
    """[summary]

    Args:
        X ([type]): [description]
        k ([type]): [description]

    Returns:
        [type]: [description]
    """
    kms = sklearn.cluster.KMeans(n_clusters=k)
    kms.fit(X)

    C = kms.cluster_centers_
    clss = kms.labels_
    return C, clss
