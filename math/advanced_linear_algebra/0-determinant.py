#!/usr/bin/env python3
"""Determinant of a Matrix"""


def determinant(matrix):
    """Calculate the determinant of an NxN matrix"""
    try:
        assert isinstance(matrix, list)
        assert all(isinstance(row, list) for row in matrix)
    except AssertionError:
        raise TypeError('matrix must be a list of lists')

    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1

    try:
        assert all(len(row) == len(matrix) for row in matrix)
    except AssertionError:
        raise ValueError('matrix must be a square matrix')

    return _determinant(matrix)


def _determinant(matrix):
    """Recursive function to calculate determinant of NxN matrix"""
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
    else:
        first_row = matrix[0]
        sub_matrices = [
            [
                [
                    row[idx2] for idx2 in range(len(row)) if idx2 != idx1
                ] for row in matrix[1:]
            ] for idx1 in range(len(first_row))
        ]
        return sum(
            (-1)**idx * first_row_element * _determinant(sub_matrix)
            for idx, (first_row_element, sub_matrix) in
            enumerate(zip(first_row, sub_matrices))
        )
