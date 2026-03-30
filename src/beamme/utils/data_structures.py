# The MIT License (MIT)
#
# Copyright (c) 2018-2026 BeamMe Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Helper functions for data structure related functionality."""

from typing import Any as _Any
from typing import Callable as _Callable

from fourcipp.utils.dict_utils import (
    compare_nested_dicts_or_lists as _compare_nested_dicts_or_lists,
)


def create_inverse_mapping(mapping: dict[_Any, _Any]) -> dict[_Any, _Any]:
    """Create an inverse mapping from a given mapping.

    Args:
        mapping: The mapping to create the inverse mapping from.

    Returns:
        The inverse mapping.
    """
    # Check that the mapping is invertible, i.e., that all values are unique.
    unique_values = set(mapping.values())
    if len(unique_values) != len(mapping):
        unique_values = set()
        duplicate_values = set()
        for value in mapping.values():
            if value in unique_values:
                duplicate_values.add(value)
            else:
                unique_values.add(value)
        raise ValueError(
            "The mapping is not invertible, values are not unique. "
            f"Non-unique values: {duplicate_values}"
        )
    return {value: key for key, value in mapping.items()}


def compare_nested_dicts_or_lists(
    obj: _Any,
    reference_obj: _Any,
    allow_int_vs_float_comparison: bool = False,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    equal_nan: bool = False,
    custom_compare: _Callable | None = None,
) -> bool:
    """Recursively compare two nested dictionaries or lists.

    This function is taken from FourCIPP and modified such that no assertion error
    is raised if the objects are not equal but instead a boolean is returned.

    To compare custom python objects, a `custom_compare` callable can be provided which:
        - Returns nothing/`None` if the objects where not compared within `custom_compare`
        - Returns `True` if the objects are seen as equal
        - Raises AssertionError if the objects are not equal

    Args:
        obj: Object for comparison
        reference_obj: Reference object
        allow_int_vs_float_comparison: Allow a tolerance based comparison between int and
            float
        rtol: The relative tolerance parameter for numpy.isclose
        atol: The absolute tolerance parameter for numpy.isclose
        equal_nan: Whether to compare NaN's as equal for numpy.isclose
        custom_compare: Callable to compare objects within this nested framework

    Returns:
        True if the dictionaries are equal
    """
    try:
        return _compare_nested_dicts_or_lists(
            obj,
            reference_obj,
            allow_int_vs_float_comparison,
            rtol,
            atol,
            equal_nan,
            custom_compare,
        )
    except AssertionError:
        return False
