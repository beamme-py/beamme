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
"""This file provides the mappings between BeamMe objects and 4C input
files."""

from typing import Any as _Any

import pyvista as _pv

from beamme.core.conf import bme as _bme
from beamme.core.mesh_representation import (
    MESH_REPRESENTATION_MAPPINGS as _MESH_REPRESENTATION_MAPPINGS,
)
from beamme.four_c.four_c_types import BeamType as _BeamType
from beamme.utils.data_structures import (
    create_inverse_mapping as _create_inverse_mapping,
)

INPUT_FILE_MAPPINGS: dict[str, _Any] = {}
INPUT_FILE_MAPPINGS["four_c_type_to_four_c_type"] = {
    _BeamType.reissner: "BEAM3R",
    _BeamType.kirchhoff: "BEAM3K",
    _BeamType.euler_bernoulli: "BEAM3EB",
}
INPUT_FILE_MAPPINGS["four_c_type_to_requires_triads"] = {
    "BEAM3R": True,
    "BEAM3K": True,
    "BEAM3EB": False,
}
INPUT_FILE_MAPPINGS["element_type_and_n_nodes_to_four_c_cell"] = {
    (_bme.element_type.beam, 2): "LINE2",
    (_bme.element_type.beam, 3): "LINE3",
    (_bme.element_type.beam, 4): "LINE4",
    (_bme.element_type.beam, 5): "LINE5",
    (_bme.element_type.nurbs, 9): "NURBS9",
    (_bme.element_type.nurbs, 27): "NURBS27",
    (_bme.element_type.solid, 8): "HEX8",
    (_bme.element_type.solid, 20): "HEX20",
    (_bme.element_type.solid, 27): "HEX27",
    (_bme.element_type.solid, 4): "TET4",
    (_bme.element_type.solid, 10): "TET10",
    (_bme.element_type.solid, 6): "WEDGE6",
    (_bme.element_type.solid, 1): "POINT1",
}
INPUT_FILE_MAPPINGS["four_c_cell_to_element_type_and_n_nodes"] = (
    _create_inverse_mapping(
        INPUT_FILE_MAPPINGS["element_type_and_n_nodes_to_four_c_cell"]
    )
)
INPUT_FILE_MAPPINGS["four_c_cell_to_vtk_cell_type_legacy"] = {
    "POINT1": _pv.CellType.VERTEX,
    "HEX8": _pv.CellType.HEXAHEDRON,
    "TET4": _pv.CellType.TETRA,
    "TET10": _pv.CellType.QUADRATIC_TETRA,
    "HEX20": _pv.CellType.QUADRATIC_HEXAHEDRON,
    "HEX27": _pv.CellType.TRIQUADRATIC_HEXAHEDRON,
    "WEDGE6": _pv.CellType.WEDGE,
}
INPUT_FILE_MAPPINGS["four_c_cell_to_vtk_connectivity_mapping"] = {
    # Only list the non-standard mappings
    "HEX20": _MESH_REPRESENTATION_MAPPINGS[
        "element_type_and_n_nodes_to_connectivity_mapping_vtk_to_beamme"
    ][(_bme.element_type.solid, 20)],
    "HEX27": _MESH_REPRESENTATION_MAPPINGS[
        "element_type_and_n_nodes_to_connectivity_mapping_vtk_to_beamme"
    ][(_bme.element_type.solid, 27)],
}
INPUT_FILE_MAPPINGS["four_c_cell_to_connectivity_mapping_from_vtk"] = {
    # Only list the non-standard mappings
    "LINE3": [0, 2, 1],
    "LINE4": [0, 3, 1, 2],
    "LINE5": [0, 4, 1, 2, 3],
    "HEX20": _MESH_REPRESENTATION_MAPPINGS[
        "element_type_and_n_nodes_to_connectivity_mapping_beamme_to_vtk"
    ][(_bme.element_type.solid, 20)],
    "HEX27": _MESH_REPRESENTATION_MAPPINGS[
        "element_type_and_n_nodes_to_connectivity_mapping_beamme_to_vtk"
    ][(_bme.element_type.solid, 27)],
}
INPUT_FILE_MAPPINGS["geometry_sets_geometry_to_entry_name"] = {
    _bme.geo.point: "DNODE",
    _bme.geo.line: "DLINE",
    _bme.geo.surface: "DSURFACE",
    _bme.geo.volume: "DVOL",
}
INPUT_FILE_MAPPINGS["boundary_conditions"] = {
    (_bme.bc.dirichlet, _bme.geo.point): "DESIGN POINT DIRICH CONDITIONS",
    (_bme.bc.dirichlet, _bme.geo.line): "DESIGN LINE DIRICH CONDITIONS",
    (_bme.bc.dirichlet, _bme.geo.surface): "DESIGN SURF DIRICH CONDITIONS",
    (_bme.bc.dirichlet, _bme.geo.volume): "DESIGN VOL DIRICH CONDITIONS",
    (_bme.bc.locsys, _bme.geo.point): "DESIGN POINT LOCSYS CONDITIONS",
    (_bme.bc.locsys, _bme.geo.line): "DESIGN LINE LOCSYS CONDITIONS",
    (_bme.bc.locsys, _bme.geo.surface): "DESIGN SURF LOCSYS CONDITIONS",
    (_bme.bc.locsys, _bme.geo.volume): "DESIGN VOL LOCSYS CONDITIONS",
    (_bme.bc.neumann, _bme.geo.point): "DESIGN POINT NEUMANN CONDITIONS",
    (_bme.bc.neumann, _bme.geo.line): "DESIGN LINE NEUMANN CONDITIONS",
    (_bme.bc.neumann, _bme.geo.surface): "DESIGN SURF NEUMANN CONDITIONS",
    (_bme.bc.neumann, _bme.geo.volume): "DESIGN VOL NEUMANN CONDITIONS",
    (
        _bme.bc.moment_euler_bernoulli,
        _bme.geo.point,
    ): "DESIGN POINT MOMENT EB CONDITIONS",
    (
        _bme.bc.beam_to_solid_volume_meshtying,
        _bme.geo.line,
    ): "BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING LINE",
    (
        _bme.bc.beam_to_solid_volume_meshtying,
        _bme.geo.volume,
    ): "BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING VOLUME",
    (
        _bme.bc.beam_to_solid_surface_meshtying,
        _bme.geo.line,
    ): "BEAM INTERACTION/BEAM TO SOLID SURFACE MESHTYING LINE",
    (
        _bme.bc.beam_to_solid_surface_meshtying,
        _bme.geo.surface,
    ): "BEAM INTERACTION/BEAM TO SOLID SURFACE MESHTYING SURFACE",
    (
        _bme.bc.beam_to_solid_surface_contact,
        _bme.geo.line,
    ): "BEAM INTERACTION/BEAM TO SOLID SURFACE CONTACT LINE",
    (
        _bme.bc.beam_to_solid_surface_contact,
        _bme.geo.surface,
    ): "BEAM INTERACTION/BEAM TO SOLID SURFACE CONTACT SURFACE",
    (_bme.bc.point_coupling, _bme.geo.point): "DESIGN POINT COUPLING CONDITIONS",
    (
        _bme.bc.beam_to_beam_contact,
        _bme.geo.line,
    ): "BEAM INTERACTION/BEAM TO BEAM CONTACT CONDITIONS",
    (
        _bme.bc.point_coupling_penalty,
        _bme.geo.point,
    ): "DESIGN POINT PENALTY COUPLING CONDITIONS",
    (
        _bme.bc.point_coupling_indirect,
        _bme.geo.line,
    ): "BEAM INTERACTION/BEAM TO BEAM POINT COUPLING CONDITIONS",
    (
        "DESIGN SURF MORTAR CONTACT CONDITIONS 3D",
        _bme.geo.surface,
    ): "DESIGN SURF MORTAR CONTACT CONDITIONS 3D",
}
INPUT_FILE_MAPPINGS["geometry_sets_geometry_to_condition_name"] = {
    _bme.geo.point: "DNODE-NODE TOPOLOGY",
    _bme.geo.line: "DLINE-NODE TOPOLOGY",
    _bme.geo.surface: "DSURF-NODE TOPOLOGY",
    _bme.geo.volume: "DVOL-NODE TOPOLOGY",
}
INPUT_FILE_MAPPINGS["geometry_sets_condition_to_geometry_name"] = (
    _create_inverse_mapping(
        INPUT_FILE_MAPPINGS["geometry_sets_geometry_to_condition_name"]
    )
)
INPUT_FILE_MAPPINGS["four_c_node_type_to_beamme_node_type"] = {
    "NODE": _bme.node_type.node,
    "CP": _bme.node_type.control_point,
}
