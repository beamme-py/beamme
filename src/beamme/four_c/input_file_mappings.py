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

from beamme.core.conf import bme as _bme
from beamme.core.element_volume import (
    VolumeHEX8 as _VolumeHEX8,
)
from beamme.core.element_volume import (
    VolumeHEX20 as _VolumeHEX20,
)
from beamme.core.element_volume import (
    VolumeHEX27 as _VolumeHEX27,
)
from beamme.core.element_volume import (
    VolumeTET4 as _VolumeTET4,
)
from beamme.core.element_volume import (
    VolumeTET10 as _VolumeTET10,
)
from beamme.core.element_volume import (
    VolumeWEDGE6 as _VolumeWEDGE6,
)
from beamme.core.nurbs_patch import NURBSSurface as _NURBSSurface
from beamme.core.nurbs_patch import NURBSVolume as _NURBSVolume
from beamme.four_c.element_volume import SolidRigidSphere as _SolidRigidSphere
from beamme.four_c.four_c_types import BeamType as _BeamType

INPUT_FILE_MAPPINGS: dict[str, _Any] = {}
INPUT_FILE_MAPPINGS["beam_types"] = {
    _BeamType.reissner: "BEAM3R",
    _BeamType.kirchhoff: "BEAM3K",
    _BeamType.euler_bernoulli: "BEAM3EB",
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
INPUT_FILE_MAPPINGS["element_type_to_four_c_string"] = {
    _VolumeHEX8: "HEX8",
    _VolumeHEX20: "HEX20",
    _VolumeHEX27: "HEX27",
    _VolumeTET4: "TET4",
    _VolumeTET10: "TET10",
    _VolumeWEDGE6: "WEDGE6",
    _SolidRigidSphere: "POINT1",
}
INPUT_FILE_MAPPINGS["element_four_c_string_to_type"] = {
    value: key
    for key, value in INPUT_FILE_MAPPINGS["element_type_to_four_c_string"].items()
}
INPUT_FILE_MAPPINGS["geometry_sets_geometry_to_condition_name"] = {
    _bme.geo.point: "DNODE-NODE TOPOLOGY",
    _bme.geo.line: "DLINE-NODE TOPOLOGY",
    _bme.geo.surface: "DSURF-NODE TOPOLOGY",
    _bme.geo.volume: "DVOL-NODE TOPOLOGY",
}
INPUT_FILE_MAPPINGS["geometry_sets_condition_to_geometry_name"] = {
    value: key
    for key, value in INPUT_FILE_MAPPINGS[
        "geometry_sets_geometry_to_condition_name"
    ].items()
}
INPUT_FILE_MAPPINGS["geometry_sets_geometry_to_entry_name"] = {
    _bme.geo.point: "DNODE",
    _bme.geo.line: "DLINE",
    _bme.geo.surface: "DSURFACE",
    _bme.geo.volume: "DVOL",
}
INPUT_FILE_MAPPINGS["n_nodes_to_cell_type"] = {
    2: "LINE2",
    3: "LINE3",
    4: "LINE4",
    5: "LINE5",
}
INPUT_FILE_MAPPINGS["n_nodes_to_cell_type_solid"] = {
    8: "HEX8",
    20: "HEX20",
    27: "HEX27",
    4: "TET4",
    10: "TET10",
    6: "WEDGE6",
    1: "POINT1",
}
INPUT_FILE_MAPPINGS["n_nodes_to_node_ordering"] = {
    2: [0, 1],
    3: [0, 2, 1],
    4: [0, 3, 1, 2],
    5: [0, 4, 1, 2, 3],
}
INPUT_FILE_MAPPINGS["nurbs_type_to_default_four_c_type"] = {
    _NURBSSurface: "WALLNURBS",
    _NURBSVolume: "SOLID",
}
