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
"""This file provides a data class to handle boundary conditions dumped to 4C input
files."""

from dataclasses import dataclass as _dataclass
from typing import Any as _Any

from beamme.core.mesh_representation import GeometrySetInfo as _GeometrySetInfo


@_dataclass
class FourCBoundaryConditionData:
    """Class that contains the data for a 4C boundary condition."""

    geometry_set_id: int
    data: dict

    def dump_to_input_file_yaml(self) -> dict:
        """Dump the boundary condition data to a dictionary that can be written to a 4C
        input file with the mesh in yaml format."""
        return {"E": self.geometry_set_id, **self.data}

    def dump_to_input_file_vtu(
        self, geometry_sets_in_mr: dict[int, _GeometrySetInfo]
    ) -> dict:
        """Dump the boundary condition data to a dictionary that can be written to a 4C
        input file with the mesh in vtu format."""
        # For geometry sets in vtu format, 4C starts counting from 0. Also, the entries in geometry_sets_in_mr are indexed from 0.
        geometry_set_id = self.geometry_set_id - 1
        geometry_set_info = geometry_sets_in_mr[geometry_set_id]

        boundary_condition_geometry_info: dict[str, _Any]
        if geometry_set_info.name is not None:
            boundary_condition_geometry_info = {"NODE_SET_NAME": geometry_set_info.name}
        else:
            boundary_condition_geometry_info = {
                "E": geometry_set_id,
                "ENTITY_TYPE": "node_set_id",
            }
        return {**boundary_condition_geometry_info, **self.data}
