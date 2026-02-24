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

from beamme.core.element import Element as _Element
from beamme.core.nurbs_patch import NURBSPatch as _NURBSPatch
from beamme.core.nurbs_patch import NURBSSurface as _NURBSSurface
from beamme.core.nurbs_patch import NURBSVolume as _NURBSVolume
from beamme.four_c.input_file_mappings import (
    INPUT_FILE_MAPPINGS as _INPUT_FILE_MAPPINGS,
)


def get_four_c_solid(solid_type: str, *, data: dict | None = None) -> type[_Element]:
    """Return an object that defines the solid element type for 4C solid
    elements."""

    if data is None:
        data = {}

    base_type: type[_NURBSPatch]
    if solid_type.startswith("nurbs_"):
        if solid_type == "nurbs_surface":
            base_type = _NURBSSurface
        elif solid_type == "nurbs_volume":
            base_type = _NURBSVolume
        else:
            raise ValueError(f"Unsupported NURBS solid type {solid_type}!")
        return type(
            "FourCSolidElementType",
            (base_type,),
            {
                "four_c_element_data": {
                    "type": _INPUT_FILE_MAPPINGS["nurbs_type_to_default_four_c_type"][
                        base_type
                    ]
                }
                | data
            },
        )
    else:
        raise ValueError(f"Unsupported solid type {solid_type}!")
