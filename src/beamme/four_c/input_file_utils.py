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
"""This module defines utility functions for an input file for 4C."""

from fourcipp.fourc_input import FourCInput as _FourCInput


def contains_yaml_based_geometry(four_cipp_input: _FourCInput) -> bool:
    """Check if the input file contains yaml-based geometry.

    Args:
        four_cipp_input: The input file to check.

    Returns:
        True if the input file contains yaml-based geometry, False otherwise.
    """
    n_yaml_nodes = len(four_cipp_input.sections.get("NODE COORDS", []))
    n_yaml_elements = sum(
        len(four_cipp_input.sections.get(section, []))
        for section in ("FLUID ELEMENTS", "STRUCTURE ELEMENTS")
    )
    return n_yaml_nodes > 0 or n_yaml_elements > 0


def get_start_index_functions(four_cipp_input: _FourCInput) -> int:
    """Get the start index for functions in the input file.

    Args:
        four_cipp_input: The input file to check.

    Returns:
        The start index for functions in the input file.
    """
    return max(
        (
            int(section.split("FUNCT")[-1])
            for section in four_cipp_input.sections
            if section.startswith("FUNCT")
        ),
        default=0,
    )
