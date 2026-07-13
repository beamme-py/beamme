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
"""This script is used to test the examples."""

from pathlib import Path

import jupytext
import pytest
from testbook import testbook


@pytest.mark.parametrize(
    "notebook_path",
    [
        Path("examples/example_1_finite_rotations.py"),
        Path("examples/example_2_core_mesh_generation_functions.py"),
    ],
)
def test_other_examples_notebooks(notebook_path, monkeypatch):
    """Parameterized test case for multiple Jupyter notebooks.

    The notebook is run and it is checked that it runs through without any
    errors/assertions.
    """
    notebook = jupytext.read(notebook_path, fmt="py:percent")

    # Run from examples/ so local helper imports work like normal script usage.
    monkeypatch.chdir(notebook_path.parent)

    with testbook(notebook) as tb:
        # execute the notebook
        tb.execute()
