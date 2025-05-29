#!/usr/bin/env python3
"""Peform matrix muliplication on list-based matrices"""


def mat_mul(mat1, mat2):
    """Multiply mat1 and mat2 in that order"""
    if len(mat1[0]) != len(mat2):
        return None
    new_mat = []
    for arr1 in mat1:
        new_row = []
        for i in range(len(mat2[0])):
            arr2 = [arr[i] for arr in mat2]
            dot = sum([x*y for x, y in zip(arr1, arr2)])
            new_row.append(dot)
        new_mat.append(new_row)
    return new_mat
