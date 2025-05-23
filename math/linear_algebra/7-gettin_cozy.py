#!/usr/bin/env python3
"""
concatenation is alike adding to a vector,
many vectors becoming one
"""


def cat_matrices2D(mat1, mat2, axis=0):

    """
    we do this ay addying two vectors
    together, with the addtion sign
    """
    if axis == 0:
        if len(mat1) and len(mat2) and len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]

    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [row1[:] + row2[:] for row1, row2 in zip(mat1, mat2)]

    return None
