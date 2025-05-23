#!/usr/bin/env python3
"""
This module defines a function to add two 2D matrices element-wise.

The function returns a new matrix with summed elements if both matrices
are the same shape. If not, it returns None.
"""


def add_matrices2D(mat1, mat2):
    """
    Adds two 2D matrices element-wise.

    Args:
        mat1 (list of list of int/float): The first 2D matrix.
        mat2 (list of list of int/float): The second 2D matrix.

    Returns:
        list of list of int/float: A new matrix containing the
        sum of corresponding
        elements from mat1 and mat2.

        None: If the input matrices do not have the same shape.

    Example:
        >>> mat1 = [[1, 2], [3, 4]]
        >>> mat2 = [[5, 6], [7, 8]]
        >>> add_matrices2D(mat1, mat2)
        [[6, 8], [10, 12]]

        >>> add_matrices2D([[1, 2]], [[1, 2, 3]])
        None
    """
    if len(mat1) != len(mat2) or any(len(row1) != len(row2)
                                     for row1, row2 in zip(mat1, mat2)):
        return None

    result = []
    for i in range(len(mat1)):
        row = []
        for a in range(len(mat1[0])):
            row.append(mat1[i][a] + mat2[i][a])
        result.append(row)
    return result
