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
"""Test environment utils."""

from unittest.mock import patch

from beamme.geometric_search.utils import (
    arborx_is_available,
    cython_is_available,
)


def test_beamme_geometric_search_utils_is_arborx_available() -> None:
    """Test is_arborx_available function."""
    with patch("beamme.geometric_search.utils._find_spec", return_value=True):
        assert arborx_is_available() is True

    with patch("beamme.geometric_search.utils._find_spec", return_value=None):
        assert arborx_is_available() is False


def test_beamme_geometric_search_utils_is_cython_available() -> None:
    """Test is_cython_available function."""
    with patch("beamme.geometric_search.utils._find_spec", return_value=True):
        assert cython_is_available() is True

    with patch("beamme.geometric_search.utils._find_spec", return_value=None):
        assert cython_is_available() is False
