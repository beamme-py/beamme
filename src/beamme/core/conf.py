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
"""This module defines a global object that manages all kind of stuff."""

from enum import Enum as _Enum
from enum import auto as _auto

# Lines to be added to each created input file
INPUT_FILE_HEADER = [
    "-" * 40,
    "This input file was created with BeamMe.",
    "Copyright (c) 2018-2025 BeamMe Authors",
    "https://beamme-py.github.io/beamme/",
    "-" * 40,
]


class Geometry(_Enum):
    """Enum for geometry types."""

    point = _auto()
    line = _auto()
    surface = _auto()
    volume = _auto()


class BoundaryCondition(_Enum):
    """Enum for boundary condition types."""

    dirichlet = _auto()
    neumann = _auto()
    locsys = _auto()
    moment_euler_bernoulli = _auto()
    beam_to_beam_contact = _auto()
    beam_to_solid_volume_meshtying = _auto()
    beam_to_solid_surface_meshtying = _auto()
    beam_to_solid_surface_contact = _auto()
    point_coupling = _auto()
    point_coupling_penalty = _auto()

    def is_point_coupling_pairwise(self) -> bool:
        """Check whether the point coupling condition should be applied
        pairwise.

        Returns:
            bool: True if the coupling should be applied individually between pairs of nodes,
                rather than to the entire geometry set as a whole.
        """
        if self == self.point_coupling:
            return False
        elif self == self.point_coupling_penalty:
            return True
        else:
            raise TypeError(f"Got unexpected coupling type: {self}")


class CouplingDofType(_Enum):
    """Enum for coupling types."""

    fix = _auto()
    joint = _auto()


class DoubleNodes(_Enum):
    """Enum for handing double nodes in Neumann conditions."""

    remove = _auto()
    keep = _auto()


class BeamMe(object):
    """A global object that stores options for the whole BeamMe application."""

    def __init__(self):
        self.set_default_values()

        # Geometry types.
        self.geo = Geometry

        # Boundary conditions types.
        self.bc = BoundaryCondition

        # Coupling types.
        self.coupling_dof = CouplingDofType

        # Handling of multiple nodes in Neumann bcs.
        self.double_nodes = DoubleNodes

    def set_default_values(self):
        """Set the configuration to the default values."""

        # Set the epsilons for comparison of different types of values.
        self.eps_quaternion = 1e-10
        self.eps_pos = 1e-10
        self.eps_knot_vector = 1e-10

        # Allow the rotation of beams when connected and the triads do not
        # match.
        self.allow_beam_rotation = True

        # Check for overlapping elements when creating an input file.
        self.check_overlapping_elements = True


bme = BeamMe()
