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
"""Unit tests for the generic NURBS creation functions."""

import pytest

from beamme.core.mesh import Mesh
from beamme.mesh_creation_functions.nurbs_generic import (
    add_geomdl_nurbs_to_mesh,
    create_geometry_sets,
)
from beamme.mesh_creation_functions.nurbs_geometries import (
    create_nurbs_brick,
    create_nurbs_flat_plate_2d,
)


@pytest.mark.parametrize(
    ("nurbs_patch", "reference_values"),
    [
        (
            create_nurbs_flat_plate_2d(1, 2, n_ele_u=1, n_ele_v=2),
            {
                "vertex_u_min_v_min": [0],
                "vertex_u_min_v_max": [9],
                "vertex_u_max_v_min": [2],
                "vertex_u_max_v_max": [11],
                "line_v_min": [0, 1, 2],
                "line_v_min_next": [3, 4, 5],
                "line_v_max_next": [6, 7, 8],
                "line_v_max": [9, 10, 11],
                "line_u_min": [0, 3, 6, 9],
                "line_u_min_next": [1, 4, 7, 10],
                "line_u_max_next": [1, 4, 7, 10],
                "line_u_max": [2, 5, 8, 11],
                "surf": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            },
        ),
        (
            create_nurbs_brick(1, 2, 3, n_ele_u=1, n_ele_v=2, n_ele_w=3),
            {
                "vertex_u_min_v_min_w_min": [0],
                "vertex_u_min_v_min_w_max": [48],
                "vertex_u_min_v_max_w_min": [9],
                "vertex_u_min_v_max_w_max": [57],
                "vertex_u_max_v_min_w_min": [2],
                "vertex_u_max_v_min_w_max": [50],
                "vertex_u_max_v_max_w_min": [11],
                "vertex_u_max_v_max_w_max": [59],
                "line_v_min_w_min": [0, 1, 2],
                "line_v_min_w_max": [48, 49, 50],
                "line_v_max_w_min": [9, 10, 11],
                "line_v_max_w_max": [57, 58, 59],
                "line_u_min_w_min": [0, 3, 6, 9],
                "line_u_min_w_max": [48, 51, 54, 57],
                "line_u_max_w_min": [2, 5, 8, 11],
                "line_u_max_w_max": [50, 53, 56, 59],
                "line_u_min_v_min": [0, 12, 24, 36, 48],
                "line_u_min_v_max": [9, 21, 33, 45, 57],
                "line_u_max_v_min": [2, 14, 26, 38, 50],
                "line_u_max_v_max": [11, 23, 35, 47, 59],
                "surf_u_min": [
                    0,
                    12,
                    24,
                    36,
                    48,
                    3,
                    15,
                    27,
                    39,
                    51,
                    6,
                    18,
                    30,
                    42,
                    54,
                    9,
                    21,
                    33,
                    45,
                    57,
                ],
                "surf_u_max": [
                    2,
                    14,
                    26,
                    38,
                    50,
                    5,
                    17,
                    29,
                    41,
                    53,
                    8,
                    20,
                    32,
                    44,
                    56,
                    11,
                    23,
                    35,
                    47,
                    59,
                ],
                "surf_v_min": [0, 12, 24, 36, 48, 1, 13, 25, 37, 49, 2, 14, 26, 38, 50],
                "surf_v_max": [
                    9,
                    21,
                    33,
                    45,
                    57,
                    10,
                    22,
                    34,
                    46,
                    58,
                    11,
                    23,
                    35,
                    47,
                    59,
                ],
                "surf_w_min": [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11],
                "surf_w_max": [48, 51, 54, 57, 49, 52, 55, 58, 50, 53, 56, 59],
                "vol": [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                    34,
                    35,
                    36,
                    37,
                    38,
                    39,
                    40,
                    41,
                    42,
                    43,
                    44,
                    45,
                    46,
                    47,
                    48,
                    49,
                    50,
                    51,
                    52,
                    53,
                    54,
                    55,
                    56,
                    57,
                    58,
                    59,
                ],
            },
        ),
    ],
)
def test_beamme_mesh_creation_functions_nurbs_generic_sets(
    get_default_test_solid_material, nurbs_patch, reference_values
):
    """Test that the add NURBS to mesh functionality returns the correct
    geometry sets."""

    # Add the nurbs to a mesh
    mesh = Mesh()
    add_geomdl_nurbs_to_mesh(
        mesh, nurbs_patch, material=get_default_test_solid_material()
    )
    nurbs_patch = mesh.elements[0]

    # Create the geometry sets for this patch
    patch_set = create_geometry_sets(nurbs_patch)

    for key, geometry_set in patch_set.items():
        set_nodes = geometry_set.get_all_nodes()
        assert len(set_nodes) == len(reference_values[key])
        for i_node, node_index in enumerate(reference_values[key]):
            assert set_nodes[i_node] is mesh.nodes[node_index]
