#!/usr/bin/env python3
"""
This module provides a simple utility to perform element-wise
addition of two arrays (lists) of numbers.

If the arrays are not the same length, the function returns None.
"""


def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise.

    Parameters:
    arr1 (list of int/float): The first array.
    arr2 (list of int/float): The second array.

    Returns:
    list of int/float: A new list where each element is the sum
                       of corresponding elements in arr1 and arr2.
    None: If arr1 and arr2 are not the same length.

    Example:
    >>> add_arrays([1, 2, 3], [4, 5, 6])
    [5, 7, 9]

    >>> add_arrays([1, 2], [1, 2, 3])
    None
    """
    if len(arr1) != len(arr2):
        return None,
    return [x+y for x, y in zip(arr1, arr2)]
