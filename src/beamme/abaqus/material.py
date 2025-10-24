# The MIT License (MIT)
#
# Copyright (c) 2018-2025 BeamMe Authors
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
"""This file provides functions to create Abaqus beam element classes."""

from beamme.core.base_mesh_item import BaseMeshItem as _BaseMeshItem


class AbaqusBeamMaterial(_BaseMeshItem):
    """A class representing an Abaqus beam material."""

    def __init__(self, name: str):
        """Initialize the material. For now it is only supported to state the
        name of the resulting element set here. The material and cross-section
        lines in the input file have to be defined manually.

        Args
        ----
        name: str
            Name of the material, this will be the name of the resulting element set
        """
        super().__init__({"name": name})

    def dump_to_list(self):
        """Return a list with the (single) item representing this material."""
        return [self.data["name"]]
