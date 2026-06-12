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
"""This file defines the base beam element."""

from typing import Any as _Any

import numpy as _np
import pyvista as _pv

from beamme.core.conf import bme as _bme
from beamme.core.element import Element as _Element


class Beam(_Element):
    """A base class for a beam element."""

    # Cell type for representing this element in vtk.
    vtk_cell_type = _pv.CellType.POLY_LINE

    # Type of this element.
    element_type = _bme.element_type.beam

    # An array that defines the parameter positions of the element nodes,
    # in ascending order.
    nodes_create: _Any = []

    def __init__(self, material=None, nodes=None):
        super().__init__(nodes=nodes, material=material)

    @classmethod
    def get_coupling_dict(cls, coupling_dof_type):
        """Return the dict to couple this beam to another beam."""

        match coupling_dof_type:
            case _bme.coupling_dof.joint:
                if cls.coupling_joint_dict is None:
                    raise ValueError(f"Joint coupling is not implemented for {cls}")
                return cls.coupling_joint_dict
            case _bme.coupling_dof.fix:
                if cls.coupling_fix_dict is None:
                    raise ValueError("Fix coupling is not implemented for {cls}")
                return cls.coupling_fix_dict
            case _:
                raise ValueError(
                    f'Coupling_dof_type "{coupling_dof_type}" is not implemented!'
                )

    def flip(self):
        """Reverse the nodes of this element.

        This is usually used when reflected.
        """
        self.nodes = [self.nodes[-1 - i] for i in range(len(self.nodes))]


def generate_beam_class(n_nodes: int):
    """Return a class representing a general beam with n_nodes in BeamMe.

    Args:
        n_nodes: Number of equally spaced nodes along the beam centerline.

    Returns:
        A beam object that has n_nodes along the centerline.
    """

    # Define the class variable responsible for creating the nodes.
    nodes_create = _np.linspace(-1, 1, num=n_nodes)

    # Create the beam class which inherits from the base beam class.
    return type(f"Beam{n_nodes}", (Beam,), {"nodes_create": nodes_create, "data": None})


Beam2 = generate_beam_class(2)
Beam3 = generate_beam_class(3)
Beam4 = generate_beam_class(4)
Beam5 = generate_beam_class(5)
