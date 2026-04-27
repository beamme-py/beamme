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
"""This module implements a basic class to manage functions in a mesh."""

from beamme.core.base_mesh_item import BaseMeshItem as _BaseMeshItem


class Function(_BaseMeshItem):
    """Holds information for a function."""

    def __init__(self, function_data):
        super().__init__(data=function_data)

        # Global index of this function in a mesh.
        self.i_global: None | int = None

    def __deepcopy__(self, memo):
        """When deepcopy is called on a mesh, we do not want the same functions
        to be copied, as this will result in multiple equal functions in the
        input file."""

        # Add this object to the memo dictionary.
        memo[id(self)] = self

        # Return this object again, as no copy should be created.
        return self
