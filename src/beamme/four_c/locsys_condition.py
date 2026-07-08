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
"""This file contains the wrapper for the LocSys condition for 4c."""

from typing import List as _List

from beamme.core.boundary_condition import BoundaryCondition as _BoundaryCondition
from beamme.core.conf import bme as _bme
from beamme.core.function import Function as _Function
from beamme.core.geometry_set import GeometrySet as _GeometrySet
from beamme.core.rotation import Rotation as _Rotation
from beamme.four_c.function_utility import (
    ensure_length_of_function_array as _ensure_length_of_function_array,
)


class LocSysCondition(_BoundaryCondition):
    """This object represents a locsys condition in 4C.

    It allows to rotate the local coordinate system used to apply Dirichlet boundary
    conditions.
    """

    def __init__(
        self,
        geometry_set: _GeometrySet,
        *,
        rotation: None | _Rotation = None,
        function_array: None | _List[_Function | int] = None,
        update_node_position: bool = False,
        use_consistent_node_normal: bool = False,
        **kwargs,
    ):
        """Initialize the object.

        Args:
            geometry_set: Geometry that this boundary condition acts on
            rotation: Object that represents the rotation of the coordinate system.
            function_array:
                - If a single function is provided, it is used to scale the entire rotation.
                - If three functions are provided, they represent the components of a rotation vector,
                  in which case no explicit rotation should be passed.
            update_node_position: Flag to enable the updated node position
            use_consistent_node_normal: Flag to use a consistent node normal
        """
        # Check for invalid input arguments
        if (
            function_array is not None
            and len(function_array) > 1
            and rotation is not None
        ):
            raise ValueError(
                "If more than a single function is provided in `function_array`, "
                "no explicit `rotation` should be given. Either provide "
                "a rotation with a single function (scaling), or three "
                "functions (rotation vector components) without rotation."
            )

        # Validate provided function array
        if function_array is None:
            function_array = [0, 0, 0]
        else:
            function_array = _ensure_length_of_function_array(function_array, 3)

        # Validate provided rotation
        if rotation is None:
            rotation_vector = [1, 1, 1]
        else:
            rotation_vector = rotation.get_rotation_vector()

        condition_dict = {
            "ROTANGLE": rotation_vector,
            "FUNCT": function_array,
            "USEUPDATEDNODEPOS": int(update_node_position),
        }

        # Append the condition string with consistent normal type for line and surface geometry
        if (
            geometry_set.geometry_type is _bme.geo.line
            or geometry_set.geometry_type is _bme.geo.surface
        ):
            condition_dict["USECONSISTENTNODENORMAL"] = int(use_consistent_node_normal)
        elif use_consistent_node_normal:
            raise ValueError(
                "The keyword use_consistent_node_normal only works for line and surface geometries."
            )

        super().__init__(
            geometry_set, data=condition_dict, bc_type=_bme.bc.locsys, **kwargs
        )
