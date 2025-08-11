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
"""This file implements beam elements for 4C."""

import warnings as _warnings

import numpy as _np

from beamme.core.conf import bme as _bme
from beamme.core.element_beam import Beam as _Beam
from beamme.core.element_beam import generate_beam_class as _generate_beam_class
from beamme.four_c.four_c_types import BeamType as _BeamType
from beamme.four_c.input_file_mappings import (
    INPUT_FILE_MAPPINGS as _INPUT_FILE_MAPPINGS,
)
from beamme.four_c.material import MaterialEulerBernoulli as _MaterialEulerBernoulli
from beamme.four_c.material import MaterialKirchhoff as _MaterialKirchhoff
from beamme.four_c.material import MaterialReissner as _MaterialReissner
from beamme.four_c.material import (
    MaterialReissnerElastoplastic as _MaterialReissnerElastoplastic,
)


def dump_four_c_beam_to_list(self) -> dict:
    """Return the dictionary representing this beam element in 4C.

    Args:
        self: The beam element to be dumped.
    """

    # Check the material.
    self._check_material()

    # Gather the element data for the input file.
    element_data = type(self).four_c_element_data | self.data
    element_data["MAT"] = self.material
    if type(self).four_c_triads:
        element_data["TRIADS"] = [
            item
            for i in _INPUT_FILE_MAPPINGS["n_nodes_to_node_ordering"][len(self.nodes)]
            for item in self.nodes[i].rotation.get_rotation_vector()
        ]

    return {
        "id": self.i_global + 1,
        "cell": {
            "type": _INPUT_FILE_MAPPINGS["n_nodes_to_cell_type"][len(self.nodes)],
            "connectivity": [
                self.nodes[i]
                for i in _INPUT_FILE_MAPPINGS["n_nodes_to_node_ordering"][
                    len(self.nodes)
                ]
            ],
        },
        "data": element_data,
    }


def get_four_c_reissner_beam(n_nodes: int, is_hermite_centerline: bool) -> type[_Beam]:
    """Return a Simo-Reissner beam for 4C."""

    four_c_element_data = {
        "type": _INPUT_FILE_MAPPINGS["beam_types"][_BeamType.reissner]
    }
    if is_hermite_centerline:
        # TODO: Move this to the four_c_element_data.
        four_c_element_data["HERMITE_CENTERLINE"] = is_hermite_centerline
        coupling_fix_dict = {"NUMDOF": 9, "ONOFF": [1, 1, 1, 1, 1, 1, 0, 0, 0]}
        coupling_joint_dict = {"NUMDOF": 9, "ONOFF": [1, 1, 1, 0, 0, 0, 0, 0, 0]}
    else:
        coupling_fix_dict = {"NUMDOF": 6, "ONOFF": [1, 1, 1, 1, 1, 1]}
        coupling_joint_dict = {"NUMDOF": 6, "ONOFF": [1, 1, 1, 0, 0, 0]}

    return type(
        "BeamFourCSimoReissner",
        (_generate_beam_class(n_nodes),),
        {
            "four_c_beam_type": _BeamType.reissner,
            "four_c_triads": True,
            "four_c_element_data": four_c_element_data,
            "valid_materials": [_MaterialReissner, _MaterialReissnerElastoplastic],
            "coupling_fix_dict": coupling_fix_dict,
            "coupling_joint_dict": coupling_joint_dict,
            "dump_to_list": dump_four_c_beam_to_list,
        },
    )


def get_four_c_kirchhoff_beam(weak=True, rotvec=True, is_fad=True) -> type[_Beam]:
    """Return a Kirchhoff-Love beam for 4C."""

    # Set the parameters for this beam.
    four_c_element_data = {
        "type": _INPUT_FILE_MAPPINGS["beam_types"][_BeamType.kirchhoff],
        "WK": weak,
        "ROTVEC": 1 if rotvec else 0,
    }
    # TODO: Move this to the four_c_element_data.
    if is_fad:
        four_c_element_data["USE_FAD"] = True

    # Show warning when not using rotvec.
    if not rotvec:
        _warnings.warn(
            "Use rotvec=False with caution, especially when applying the boundary conditions "
            "and couplings."
        )

    coupling_fix_dict = {"NUMDOF": 7, "ONOFF": [1, 1, 1, 1, 1, 1, 0]}
    coupling_joint_dict = {"NUMDOF": 7, "ONOFF": [1, 1, 1, 0, 0, 0, 0]}

    return type(
        "BeamFourCKirchhoffLove",
        (_generate_beam_class(3),),
        {
            "four_c_beam_type": _BeamType.kirchhoff,
            "four_c_triads": True,
            "four_c_element_data": four_c_element_data,
            "valid_materials": [_MaterialKirchhoff],
            "coupling_fix_dict": coupling_fix_dict,
            "coupling_joint_dict": coupling_joint_dict,
            "dump_to_list": dump_four_c_beam_to_list,
        },
    )


class BeamFourCEulerBernoulli(_generate_beam_class(2)):  # type: ignore[misc]
    """Represents a Euler Bernoulli beam element."""

    four_c_beam_type = _BeamType.euler_bernoulli
    four_c_triads = False
    four_c_element_data = {
        "type": _INPUT_FILE_MAPPINGS["beam_types"][_BeamType.euler_bernoulli]
    }

    valid_materials = [_MaterialEulerBernoulli]

    def dump_to_list(self):
        """Return a list with the (single) item representing this element."""

        # Check the material.
        self._check_material()

        # The two rotations must be the same and the x1 vector must point from
        # the start point to the end point.
        if not self.nodes[0].rotation == self.nodes[1].rotation:
            raise ValueError(
                "The two nodal rotations in Euler Bernoulli beams must be the same, i.e. the beam "
                "has to be straight!"
            )
        direction = self.nodes[1].coordinates - self.nodes[0].coordinates
        t1 = self.nodes[0].rotation * [1, 0, 0]
        if _np.linalg.norm(direction / _np.linalg.norm(direction) - t1) >= _bme.eps_pos:
            raise ValueError(
                "The rotations do not match the direction of the Euler Bernoulli beam!"
            )

        return dump_four_c_beam_to_list(self)


def get_four_c_beam(
    beam_type: _BeamType,
    *,
    n_nodes: int | None = None,
    is_hermite_centerline: bool | None = None,
    **kwargs,
) -> type[_Beam]:
    """Return an object that can be used to create beams with 4C."""

    def _check_arguments(name, value, expected):
        """Check that if an argument is given, it has the expected value."""
        if value is not None and not value == expected:
            raise ValueError(
                f"Parameter {name} with the value {value} does not match the expected value {expected}"
            )

    match beam_type:
        case _BeamType.reissner:
            # Set default values for centerline interpolation
            if n_nodes is None:
                n_nodes = 3
            if is_hermite_centerline is None:
                is_hermite_centerline = True
            return get_four_c_reissner_beam(
                n_nodes=n_nodes, is_hermite_centerline=is_hermite_centerline, **kwargs
            )
        case _BeamType.kirchhoff:
            _check_arguments("n_nodes", n_nodes, 3)
            _check_arguments("is_hermite_centerline", is_hermite_centerline, True)
            return get_four_c_kirchhoff_beam(**kwargs)
        case _BeamType.euler_bernoulli:
            _check_arguments("n_nodes", n_nodes, 2)
            _check_arguments("is_hermite_centerline", is_hermite_centerline, True)
            return BeamFourCEulerBernoulli
        case _:
            raise ValueError("Got unexpected beam type.")


# Provide shortcuts for backwards compatibility
Beam3rHerm2Line3 = get_four_c_beam(
    _BeamType.reissner, n_nodes=3, is_hermite_centerline=True
)
Beam3rLine2Line2 = get_four_c_beam(
    _BeamType.reissner, n_nodes=2, is_hermite_centerline=False
)
Beam3eb = get_four_c_beam(_BeamType.euler_bernoulli)
