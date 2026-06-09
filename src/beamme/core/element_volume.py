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
"""This file defines the base volume element."""

import pyvista as _pv

from beamme.core.conf import bme as _bme
from beamme.core.element import Element as _Element


class VolumeElement(_Element):
    """A base class for a volume element."""

    # This class variables stores the information about the element shape in vtk.
    vtk_cell_type = None

    # Type of this element.
    element_type = _bme.element_type.solid

    def __init__(self, nodes=None, data={}, **kwargs):
        super().__init__(nodes=nodes, **kwargs)
        self.data = data


class VolumePoint(VolumeElement):
    """A point volume element, e.g., for spheres."""

    vtk_cell_type = _pv.CellType.VERTEX


class VolumeWEDGE6(VolumeElement):
    """A WEDGE6 volume element."""

    vtk_cell_type = _pv.CellType.WEDGE


class VolumeHEX8(VolumeElement):
    """A HEX8 volume element."""

    vtk_cell_type = _pv.CellType.HEXAHEDRON


class VolumeTET4(VolumeElement):
    """A TET4 volume element."""

    vtk_cell_type = _pv.CellType.TETRA


class VolumeTET10(VolumeElement):
    """A TET10 volume element."""

    vtk_cell_type = _pv.CellType.QUADRATIC_TETRA


class VolumeHEX20(VolumeElement):
    """A HEX20 volume element."""

    vtk_cell_type = _pv.CellType.QUADRATIC_HEXAHEDRON


class VolumeHEX27(VolumeElement):
    """A HEX27 volume element."""

    vtk_cell_type = _pv.CellType.TRIQUADRATIC_HEXAHEDRON
