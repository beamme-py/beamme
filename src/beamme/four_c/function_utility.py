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
"""This module implements utility functions to create 4C space time function."""

from typing import List as _List

import numpy as _np

from beamme.core.function import Function as _Function


def create_linear_interpolation_dict(
    times: _List[float], values: _List[float], *, variable_name="var", variable_index=0
):
    """Create a string that describes a variable that is linear interpolated over time.

    Args
        times, values:
            Time and values that will be interpolated with piecewise linear functions
        variable_name:
            Name of the created variable
        variable_index:
            Index of this created variable
    """
    if not len(times) == len(values):
        raise ValueError(
            f"The dimensions of time ({len(times)}) and values ({len(values)}) do not match"
        )

    times_appended = _np.array(times)
    values_appended = _np.array(values)
    t_max = _np.max(times_appended)
    times_appended = _np.insert(times_appended, 0, -1000.0, axis=0)
    times_appended = _np.append(times_appended, [t_max + 1000.0])
    values_appended = _np.insert(values_appended, 0, values[0], axis=0)
    values_appended = _np.append(values_appended, values_appended[-1])
    return {
        "VARIABLE": variable_index,
        "NAME": variable_name,
        "TYPE": "linearinterpolation",
        "NUMPOINTS": len(times_appended),
        "TIMES": times_appended.tolist(),
        "VALUES": values_appended.tolist(),
    }


def create_linear_interpolation_function(
    times: _List[float],
    values: _List[float],
    *,
    function_type="SYMBOLIC_FUNCTION_OF_SPACE_TIME",
):
    """Create a function that describes a linear interpolation between the given time
    points and values. Before and after it will be constant.

    Args
    ----
        times, values:
            Time and values that will be interpolated with piecewise linear functions
    """
    function_dict = create_linear_interpolation_dict(times, values, variable_name="var")
    return _Function([{function_type: "var"}, function_dict])


def ensure_length_of_function_array(function_array: _List, length: int = 3):
    """Performs size check of a function array and appends the function array to the
    given length, if a list with only one item is provided.

    Args:
        function_array: list with functions
        length: expected length of function array

    Returns:
        function_array: list with functions with provided length
    """
    # extend items of function automatically if it is only provided once
    if len(function_array) == 1:
        function_array = function_array * length

    if len(function_array) != length:
        raise ValueError(
            f"The function array must have length {length} not {len(function_array)}."
        )
    return function_array
