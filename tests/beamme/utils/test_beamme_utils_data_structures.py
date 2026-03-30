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
"""This script is used to unit test the data structure utility functions."""

import pytest

from beamme.utils.data_structures import create_inverse_mapping


def test_beamme_utils_data_structures_create_inverse_mapping():
    """Test the create_inverse_mapping function."""

    # Test with a simple mapping.
    mapping = {1: "a", 2: "b", 3: "c"}
    inverse_mapping = create_inverse_mapping(mapping)
    assert inverse_mapping == {"a": 1, "b": 2, "c": 3}

    # Test with a mapping that has duplicate values.
    mapping = {1: "a", 2: "b", 3: "a"}
    with pytest.raises(
        ValueError,
        match="The mapping is not invertible, values are not unique. Non-unique values: {'a'}",
    ):
        create_inverse_mapping(mapping)
