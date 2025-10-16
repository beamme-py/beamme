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
"""This file defines functions to dump mesh items for 4C."""

from typing import Any as _Any

from beamme.core.boundary_condition import BoundaryCondition as _BoundaryCondition
from beamme.core.conf import bme as _bme
from beamme.core.coupling import Coupling as _Coupling
from beamme.core.element_volume import VolumeElement as _VolumeElement
from beamme.core.geometry_set import GeometrySet as _GeometrySet
from beamme.core.geometry_set import GeometrySetNodes as _GeometrySetNodes
from beamme.core.node import ControlPoint as _ControlPoint
from beamme.core.node import Node as _Node
from beamme.core.nurbs_patch import NURBSPatch as _NURBSPatch
from beamme.four_c.four_c_types import BeamType as _BeamType
from beamme.four_c.input_file_mappings import (
    INPUT_FILE_MAPPINGS as _INPUT_FILE_MAPPINGS,
)


def dump_node(node):
    """Return the representation of a node in the 4C input file."""

    if isinstance(node, _ControlPoint):
        return {
            "id": node.i_global + 1,
            "COORD": node.coordinates,
            "data": {"type": "CP", "weight": node.weight},
        }
    elif isinstance(node, _Node):
        return {
            "id": node.i_global + 1,
            "COORD": node.coordinates,
            "data": {"type": "NODE"},
        }
    else:
        raise TypeError(f"Got unexpected item of type {type(node)}")


def dump_solid_element(solid_element):
    """Return a dict with the items representing the given solid element."""

    return {
        "id": solid_element.i_global + 1,
        "cell": {
            "type": _INPUT_FILE_MAPPINGS["element_type_to_four_c_string"][
                type(solid_element)
            ],
            "connectivity": solid_element.nodes,
        },
        "data": solid_element.data,
    }


def dump_coupling(coupling):
    """Return the input file representation of the coupling condition."""

    if isinstance(coupling.data, dict):
        data = coupling.data
    else:
        # In this case we have to check which beams are connected to the node.
        # TODO: Coupling also makes sense for different beam types, this can
        # be implemented at some point.
        nodes = coupling.geometry_set.get_points()
        connected_elements = [
            element for node in nodes for element in node.element_link
        ]
        element_types = {type(element) for element in connected_elements}
        if len(element_types) > 1:
            raise TypeError(
                f"Expected a single connected type of beam elements, got {element_types}"
            )
        element_type = element_types.pop()
        if element_type.four_c_beam_type is _BeamType.kirchhoff:
            rotvec = {
                type(element).four_c_element_data["ROTVEC"]
                for element in connected_elements
            }
            if len(rotvec) > 1 or not rotvec.pop():
                raise TypeError(
                    "Couplings for Kirchhoff beams and rotvec==False not yet implemented."
                )

        data = element_type.get_coupling_dict(coupling.data)

    return {"E": coupling.geometry_set.i_global + 1, **data}


def dump_geometry_set(geometry_set):
    """Return a list with the data describing this set."""

    # Sort nodes based on their global index
    nodes = sorted(geometry_set.get_all_nodes(), key=lambda n: n.i_global)

    if not nodes:
        raise ValueError("Writing empty geometry sets is not supported")

    return [
        {
            "type": "NODE",
            "node_id": node.i_global + 1,
            "d_type": _INPUT_FILE_MAPPINGS["geometry_sets_geometry_to_entry_name"][
                geometry_set.geometry_type
            ],
            "d_id": geometry_set.i_global + 1,
        }
        for node in nodes
    ]


def dump_nurbs_patch_knotvectors(input_file, nurbs_patch) -> None:
    """Set the knot vectors of the NURBS patch in the input file."""

    patch_data: dict[str, _Any] = {
        "KNOT_VECTORS": [],
    }

    for dir_manifold in range(nurbs_patch.get_nurbs_dimension()):
        knotvector = nurbs_patch.knot_vectors[dir_manifold]
        num_knots = len(knotvector)

        # Check the type of knot vector, in case that the multiplicity of the first and last
        # knot vectors is not p + 1, then it is a closed (periodic) knot vector, otherwise it
        # is an open (interpolated) knot vector.
        knotvector_type = "Interpolated"

        for i in range(nurbs_patch.polynomial_orders[dir_manifold] - 1):
            if (abs(knotvector[i] - knotvector[i + 1]) > _bme.eps_knot_vector) or (
                abs(knotvector[num_knots - 2 - i] - knotvector[num_knots - 1 - i])
                > _bme.eps_knot_vector
            ):
                knotvector_type = "Periodic"
                break

        patch_data["KNOT_VECTORS"].append(
            {
                "DEGREE": nurbs_patch.polynomial_orders[dir_manifold],
                "TYPE": knotvector_type,
                "KNOTS": [
                    knot_vector_val
                    for knot_vector_val in nurbs_patch.knot_vectors[dir_manifold]
                ],
            }
        )

    if "STRUCTURE KNOTVECTORS" in input_file:
        # Get all existing patches in the input file - they will be added to the
        # input file again at the end of this function. By doing it this way, the
        # FourCIPP type converter will be applied to the current patch.
        # This also means that we apply the type converter again already existing
        # patches. But, with the usual number of patches and data size, this
        # should not lead to a measurable performance impact.
        patches = input_file.pop("STRUCTURE KNOTVECTORS")["PATCHES"]
    else:
        patches = []

    patch_data["ID"] = nurbs_patch.i_nurbs_patch + 1
    patches.append(patch_data)
    input_file.add({"STRUCTURE KNOTVECTORS": {"PATCHES": patches}})


def dump_nurbs_patch_elements(nurbs_patch: _NURBSPatch) -> list[dict[str, _Any]]:
    """Return a list with all the element definitions contained in this
    patch."""

    if nurbs_patch.i_global is None:
        raise ValueError(
            "i_global is not set, make sure that the NURBS patch is added to the mesh"
        )

    # Check the material
    nurbs_patch._check_material()

    patch_elements = []
    j = 0

    for knot_span in nurbs_patch.get_knot_span_iterator():
        element_cps_ids = nurbs_patch.get_ids_ctrlpts(*knot_span)
        connectivity = [nurbs_patch.nodes[i] for i in element_cps_ids]
        num_cp = len(connectivity)

        patch_elements.append(
            {
                "id": nurbs_patch.i_global + j + 1,
                "cell": {
                    "type": f"NURBS{num_cp}",
                    "connectivity": connectivity,
                },
                "data": {
                    "type": _INPUT_FILE_MAPPINGS["nurbs_type_to_default_four_c_type"][
                        type(nurbs_patch)
                    ],
                    "MAT": nurbs_patch.material,
                    **(nurbs_patch.data if nurbs_patch.data else {}),
                },
            }
        )
        j += 1

    return patch_elements


def dump_item_to_list(dumped_list, item) -> None:
    """General function to dump items to a 4C input file."""
    if hasattr(item, "dump_to_list"):
        dumped_list.append(item.dump_to_list())
    elif isinstance(item, _Node):
        dumped_list.append(dump_node(item))
    elif isinstance(item, _VolumeElement):
        dumped_list.append(dump_solid_element(item))
    elif isinstance(item, _GeometrySet) or isinstance(item, _GeometrySetNodes):
        dumped_list.extend(dump_geometry_set(item))
    elif isinstance(item, _NURBSPatch):
        dumped_list.extend(dump_nurbs_patch_elements(item))
    elif isinstance(item, _BoundaryCondition):
        if item.geometry_set.i_global is None:
            raise ValueError("i_global is not set")
        dumped_list.append(
            {
                "E": item.geometry_set.i_global + 1,
                **item.data,
            }
        )
    elif isinstance(item, _Coupling):
        dumped_list.append(dump_coupling(item))
    else:
        raise TypeError(f"Could not dump {item}")


def dump_item_to_section(input_file, item) -> None:
    """This function dumps information of mesh items to general input file
    sections, e.g., knotvectors for NURBS."""
    if isinstance(item, _NURBSPatch):
        dump_nurbs_patch_knotvectors(input_file, item)
