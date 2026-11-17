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
"""Unit tests the 4C input file utils."""

from dataclasses import dataclass

import pytest
from fourcipp.fourc_input import FourCInput

from beamme.four_c.input_file_utils import (
    contains_yaml_based_geometry,
    four_cipp_unregister_type,
)


def test_beamme_four_c_input_file_utils_contains_yaml_based_geometry():
    """Test that the function correctly identifies yaml-based geometry."""
    nodes = [
        {"id": 1, "COORD": [0.0, 0.0, 0.0], "data": {"type": "NODE"}},
        {"id": 2, "COORD": [1.0, 0.0, 0.0], "data": {"type": "NODE"}},
    ]
    elements = [
        {
            "id": 1,
            "cell": {"type": "TET4", "connectivity": [1, 2, 3, 4]},
            "data": {"type": "SOLID", "MAT": 1, "KINEM": "nonlinear"},
        },
        {
            "id": 2,
            "cell": {"type": "TET4", "connectivity": [2, 3, 4, 5]},
            "data": {"type": "SOLID", "MAT": 1, "KINEM": "nonlinear"},
        },
    ]

    def get_input_file() -> FourCInput:
        """Return an input file with an arbitrary parameter."""
        input_file = FourCInput()
        input_file["BEAM INTERACTION"] = {"REPARTITIONSTRATEGY": "adaptive"}
        return input_file

    # Check empty input file
    input_file = get_input_file()
    assert not contains_yaml_based_geometry(input_file)

    # Check input file with only nodes
    input_file["NODE COORDS"] = nodes
    assert contains_yaml_based_geometry(input_file)

    # Check input file with only elements
    input_file = get_input_file()
    input_file["STRUCTURE ELEMENTS"] = elements
    assert contains_yaml_based_geometry(input_file)

    # Check input file with nodes and elements
    input_file = get_input_file()
    input_file["NODE COORDS"] = nodes
    input_file["STRUCTURE ELEMENTS"] = elements
    assert contains_yaml_based_geometry(input_file)


def test_beamme_four_c_input_file_utils_four_cipp_unregister_type():
    """Test the four_cipp_unregister_type function."""

    @dataclass
    class Dummy:
        """Data class to be converted."""

        data: int

    dummy_data = Dummy(data=42)

    input_file = FourCInput()

    input_file.type_converter.register_type(Dummy, lambda _, obj: obj.data)
    assert input_file.type_converter(dummy_data) == 42

    four_cipp_unregister_type(input_file, Dummy)

    with pytest.raises(
        ValueError, match=r"Custom type .* is not registered in the FourCInput object\."
    ):
        four_cipp_unregister_type(input_file, Dummy)
