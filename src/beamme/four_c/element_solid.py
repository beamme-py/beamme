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
"""This file implements solid elements for 4C."""

import copy as _copy

from beamme.core.conf import ElementType as _ElementType
from beamme.core.element import Element as _Element
from beamme.core.element_volume import VolumeHEX8 as _VolumeHEX8
from beamme.core.element_volume import VolumeHEX20 as _VolumeHEX20
from beamme.core.element_volume import VolumeHEX27 as _VolumeHEX27
from beamme.core.element_volume import VolumePoint as _VolumePoint
from beamme.core.element_volume import VolumeTET4 as _VolumeTET4
from beamme.core.element_volume import VolumeTET10 as _VolumeTET10
from beamme.core.element_volume import VolumeWEDGE6 as _VolumeWEDGE6
from beamme.core.nurbs_patch import NURBSSurface as _NURBSSurface
from beamme.core.nurbs_patch import NURBSVolume as _NURBSVolume
from beamme.four_c.input_file_mappings import (
    INPUT_FILE_MAPPINGS as _INPUT_FILE_MAPPINGS,
)
from beamme.four_c.material import MaterialSolid as _MaterialSolid


def get_four_c_solid(
    solid_type: _ElementType,
    four_c_type: str,
    n_nodes: int,
    *,
    element_technology: dict | None = None,
) -> type[_Element]:
    """Return a type that defines a solid element block for 4C solid elements.

    Args:
        solid_type: Type of solid element. Supported values are:
            - solid
            - nurbs
        four_c_type: The 4C type of the solid element.
        n_nodes: The number of nodes of the solid element.
        element_technology: Dictionary specifying element technologies. Will be deep copied.

    Returns:
        A type that defines a solid element block for 4C solid elements.
    """

    if element_technology is None:
        element_technology = {}
    else:
        element_technology = _copy.deepcopy(element_technology)

    base_type: type[_Element]

    match solid_type:
        case _ElementType.nurbs:
            match n_nodes:
                case 9:
                    base_type = _NURBSSurface
                case 27:
                    base_type = _NURBSVolume
                case _:
                    raise ValueError(
                        f"Unsupported number of nodes {n_nodes} for NURBS element!"
                    )
        case _ElementType.solid:
            match n_nodes:
                case 1:
                    base_type = _VolumePoint
                case 8:
                    base_type = _VolumeHEX8
                case 20:
                    base_type = _VolumeHEX20
                case 27:
                    base_type = _VolumeHEX27
                case 4:
                    base_type = _VolumeTET4
                case 10:
                    base_type = _VolumeTET10
                case 6:
                    base_type = _VolumeWEDGE6
                case _:
                    raise ValueError(
                        f"Unsupported number of nodes {n_nodes} for solid element!"
                    )
        case _:
            raise ValueError(f"Unsupported solid type {solid_type}!")

    four_c_cell = _INPUT_FILE_MAPPINGS["element_type_and_n_nodes_to_four_c_cell"][
        solid_type, n_nodes
    ]
    # All elements, except for the point element (i.e., rigid sphere), require a solid material.
    valid_materials = [_MaterialSolid] if base_type is not _VolumePoint else None
    return type(
        "FourCSolidElementType",
        (base_type,),
        {
            "element_type": solid_type,
            "data": {four_c_type: {four_c_cell: element_technology}},
            "valid_materials": valid_materials,
        },
    )
