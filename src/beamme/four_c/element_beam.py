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
"""This file implements beam elements for 4C."""

import warnings as _warnings
from typing import Any as _Any

import numpy as _np

from beamme.core.conf import bme as _bme
from beamme.core.element_beam import Beam as _Beam
from beamme.core.element_beam import Beam2 as _Beam2
from beamme.core.element_beam import generate_beam_class as _generate_beam_class
from beamme.four_c.four_c_types import (
    BeamKirchhoffConstraintType as _BeamKirchhoffConstraintType,
)
from beamme.four_c.four_c_types import (
    BeamKirchhoffParametrizationType as _BeamKirchhoffParametrizationType,
)
from beamme.four_c.four_c_types import BeamType as _BeamType
from beamme.four_c.input_file_mappings import (
    INPUT_FILE_MAPPINGS as _INPUT_FILE_MAPPINGS,
)


def get_four_c_reissner_beam(n_nodes: int, is_hermite_centerline: bool) -> type[_Beam]:
    """Return a Simo-Reissner beam for 4C."""

    four_c_type = _INPUT_FILE_MAPPINGS["four_c_type_to_four_c_type"][_BeamType.reissner]
    four_c_cell = _INPUT_FILE_MAPPINGS["element_type_and_n_nodes_to_four_c_cell"][
        _bme.element_type.beam, n_nodes
    ]
    element_technology = {"HERMITE_CENTERLINE": is_hermite_centerline}

    if is_hermite_centerline:
        coupling_fix_dict = {"NUMDOF": 9, "ONOFF": [1, 1, 1, 1, 1, 1, 0, 0, 0]}
        coupling_joint_dict = {"NUMDOF": 9, "ONOFF": [1, 1, 1, 0, 0, 0, 0, 0, 0]}
    else:
        coupling_fix_dict = {"NUMDOF": 6, "ONOFF": [1, 1, 1, 1, 1, 1]}
        coupling_joint_dict = {"NUMDOF": 6, "ONOFF": [1, 1, 1, 0, 0, 0]}

    return type(
        "BeamFourCSimoReissner",
        (_generate_beam_class(n_nodes),),
        {
            "element_type": _bme.element_type.beam,
            "beam_type": _BeamType.reissner,
            "data": {four_c_type: {four_c_cell: element_technology}},
            "coupling_fix_dict": coupling_fix_dict,
            "coupling_joint_dict": coupling_joint_dict,
        },
    )


def get_four_c_kirchhoff_beam(
    constraint: _BeamKirchhoffConstraintType = _BeamKirchhoffConstraintType.weak,
    parametrization: _BeamKirchhoffParametrizationType = _BeamKirchhoffParametrizationType.rot,
    is_fad: bool = True,
) -> type[_Beam]:
    """Return a Kirchhoff-Love beam for 4C."""

    # Show warning when not using rotvec.
    if not parametrization == _BeamKirchhoffParametrizationType.rot:
        _warnings.warn(
            "Use tangent based parametrization with caution, especially when "
            " applying the boundary conditions and couplings."
        )

    n_nodes = 3
    four_c_type = _INPUT_FILE_MAPPINGS["four_c_type_to_four_c_type"][
        _BeamType.kirchhoff
    ]
    four_c_cell = _INPUT_FILE_MAPPINGS["element_type_and_n_nodes_to_four_c_cell"][
        _bme.element_type.beam, n_nodes
    ]

    element_technology = {
        "CONSTRAINT": constraint.name,
        "PARAMETRIZATION": parametrization.name,
        "USE_FAD": is_fad,
    }

    coupling_fix_dict = {"NUMDOF": 7, "ONOFF": [1, 1, 1, 1, 1, 1, 0]}
    coupling_joint_dict = {"NUMDOF": 7, "ONOFF": [1, 1, 1, 0, 0, 0, 0]}

    return type(
        "BeamFourCKirchhoffLove",
        (_generate_beam_class(n_nodes),),
        {
            "element_type": _bme.element_type.beam,
            "beam_type": _BeamType.kirchhoff,
            "kirchhoff_parametrization": parametrization,
            "data": {four_c_type: {four_c_cell: element_technology}},
            "coupling_fix_dict": coupling_fix_dict,
            "coupling_joint_dict": coupling_joint_dict,
        },
    )


class BeamFourCEulerBernoulli(_Beam2):
    """Represents a Euler Bernoulli beam element."""

    element_type = _bme.element_type.beam
    beam_type = _BeamType.euler_bernoulli
    data: dict[str, dict[str, _Any]] = {
        _INPUT_FILE_MAPPINGS["four_c_type_to_four_c_type"][_BeamType.euler_bernoulli]: {
            _INPUT_FILE_MAPPINGS["element_type_and_n_nodes_to_four_c_cell"][
                _bme.element_type.beam, len(_Beam2.nodes_create)
            ]: {}
        }
    }

    def check(self) -> None:
        """Check that the beam is straight and that the two rotations are the
        same."""

        # Perform checks from the parent class.
        super().check()

        # The two rotations must be the same and the x1 vector must point from
        # the start point to the end point.
        if not self.nodes[0].rotation == self.nodes[1].rotation:
            raise ValueError(
                "The two nodal rotations in Euler Bernoulli beams must be the same,"
                "i.e., the beam has to be straight!"
            )
        direction = self.nodes[1].coordinates - self.nodes[0].coordinates
        t1 = self.nodes[0].rotation * [1, 0, 0]
        if _np.linalg.norm(direction / _np.linalg.norm(direction) - t1) >= _bme.eps_pos:
            raise ValueError(
                "The rotations do not match the direction of the Euler Bernoulli beam!"
            )


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
