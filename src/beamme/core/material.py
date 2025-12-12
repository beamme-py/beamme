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
"""This file implements basic classes to manage materials."""

import numpy as _np

from beamme.core.base_mesh_item import BaseMeshItem as _BaseMeshItem


class Material(_BaseMeshItem):
    """Base class for all materials."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __deepcopy__(self, memo):
        """When deepcopy is called on a mesh, we do not want the materials to
        be copied, as this will result in multiple equal materials in the input
        file."""

        # Add this object to the memo dictionary.
        memo[id(self)] = self

        # Return this object again, as no copy should be created.
        return self

    def get_all_contained_materials(self) -> list:
        """Get all sub materials contained in this material, also nested ones.

        Returns:
            List of all contained materials, for the base class this is empty.
        """
        return []


class MaterialBeamBase(Material):
    """Base class for all beam materials."""

    def __init__(
        self,
        radius=None,
        material_string=None,
        youngs_modulus=None,
        nu=0.0,
        density=0.0,
        interaction_radius=None,
        **kwargs,
    ):
        """Set the material values that all beams have."""

        super().__init__(**kwargs)

        self.radius = radius
        self.material_string = material_string
        self.youngs_modulus = youngs_modulus
        self.nu = nu
        self.density = density
        self.interaction_radius = interaction_radius
        self.area = None
        self.mom2 = None
        self.mom3 = None
        self.polar = None

    def calc_area_stiffness(self):
        """Calculate the relevant stiffness terms and the area for the given
        beam."""
        area = 4 * self.radius**2 * _np.pi * 0.25
        mom2 = self.radius**4 * _np.pi * 0.25
        mom3 = mom2
        polar = mom2 + mom3
        return area, mom2, mom3, polar


class MaterialSolidBase(Material):
    """Base class for all solid materials."""

    pass
