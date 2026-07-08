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

import pytest
from testbook import testbook


@pytest.mark.parametrize(
    "notebook_path",
    [
        "examples/example_1_finite_rotations.ipynb",
        "examples/example_2_core_mesh_generation_functions.ipynb",
    ],
)
def test_other_examples_notebooks(notebook_path):
    """Parameterized test case for multiple Jupyter notebooks.

    The notebook is run and it is checked that it runs through without any
    errors/assertions.
    """
    with testbook(notebook_path) as tb:
        # we do not define the examples as modules, therefore we need to add the
        # examples folder to the current sys path so examples/utils can be imported
        # within the notebooks correctly
        tb.inject(
            """
            import sys
            import os
            sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "examples")))
            """
        )

        # execute the notebook
        tb.execute()
