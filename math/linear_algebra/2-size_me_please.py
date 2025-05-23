#!/usr/bin/env python3

"""
Module for returning the shape or dimension of any given matrix.

Given a matrix, returns the shape of it as a list of integers,
where each integer represents the size of a corresponding dimension.
"""


def matrix_shape(matrix):
    """
    Given a matrix (list of lists), returns the shape of the matrix
    as a list of integers representing the size of each dimension.

    Parameters:
    matrix (list): The matrix whose shape is to be determined.

    Returns:
    list: A list of integers where each integer corresponds to the
          size of a dimension of the matrix.
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0] if matrix else []
    return shape
