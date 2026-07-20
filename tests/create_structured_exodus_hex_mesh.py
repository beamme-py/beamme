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
"""Utilities to generate simple structured hexahedral meshes and write them to Exodus
(netCDF) files for testing."""

from pathlib import Path

import netCDF4
import numpy as np
from cubitpy.conf import cupy
from cubitpy.cubit_utility import node_set_info_to_string
from numpy.typing import NDArray

from beamme.core.conf import bme
from beamme.core.mesh_representation import MESH_REPRESENTATION_MAPPINGS
from beamme.four_c.input_file import InputFile


def create_structured_hex_mesh(
    nx: int,
    ny: int,
    nz: int,
    *,
    element: str = "hex8",
    bounds: tuple | list = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
) -> tuple[NDArray, NDArray]:
    """Create a structured hexahedral mesh with the given number of elements in each
    direction.

    The node numbering follows the one used in 4C.

    Args:
        nx: Number of elements in x direction.
        ny: Number of elements in y direction.
        nz: Number of elements in z direction.
        element: "hex8" or "hex27" for the type of hexahedral element.
        bounds: Tuple of (min, max) for each coordinate direction.

    Returns:
        points: Array of shape (num_points, 3) with the coordinates of the points.
        connectivity: Array of shape (num_elements, nodes_per_element) with the point indices for each element.
    """
    if element == "hex8":
        px, py, pz = nx + 1, ny + 1, nz + 1
        nodes_per_elem = 8
        step = 1
    elif element == "hex27":
        px, py, pz = 2 * nx + 1, 2 * ny + 1, 2 * nz + 1
        nodes_per_elem = 27
        step = 2
    else:
        raise ValueError("element must be 'hex8' or 'hex27'")

    x = np.linspace(*bounds[0], px)
    y = np.linspace(*bounds[1], py)
    z = np.linspace(*bounds[2], pz)

    points_x, points_y, points_z = np.meshgrid(x, y, z, indexing="ij")
    points = np.column_stack((points_x.ravel(), points_y.ravel(), points_z.ravel()))

    def idx(i, j, k):
        """Get the node index for the given spatial counters i, j and k."""
        return i * py * pz + j * pz + k

    connectivity = np.empty((nx * ny * nz, nodes_per_elem), dtype=int)

    for i_x in range(nx):
        for i_y in range(ny):
            for i_z in range(nz):
                point_index_x = step * i_x
                point_index_y = step * i_y
                point_index_z = step * i_z

                if element == "hex8":
                    conn = [
                        idx(point_index_x, point_index_y, point_index_z),
                        idx(point_index_x + 1, point_index_y, point_index_z),
                        idx(point_index_x + 1, point_index_y + 1, point_index_z),
                        idx(point_index_x, point_index_y + 1, point_index_z),
                        idx(point_index_x, point_index_y, point_index_z + 1),
                        idx(point_index_x + 1, point_index_y, point_index_z + 1),
                        idx(point_index_x + 1, point_index_y + 1, point_index_z + 1),
                        idx(point_index_x, point_index_y + 1, point_index_z + 1),
                    ]

                elif element == "hex27":
                    conn = [
                        # corners
                        idx(point_index_x, point_index_y, point_index_z),
                        idx(point_index_x + 2, point_index_y, point_index_z),
                        idx(point_index_x + 2, point_index_y + 2, point_index_z),
                        idx(point_index_x, point_index_y + 2, point_index_z),
                        idx(point_index_x, point_index_y, point_index_z + 2),
                        idx(point_index_x + 2, point_index_y, point_index_z + 2),
                        idx(point_index_x + 2, point_index_y + 2, point_index_z + 2),
                        idx(point_index_x, point_index_y + 2, point_index_z + 2),
                        # edge midpoints
                        idx(point_index_x + 1, point_index_y, point_index_z),
                        idx(point_index_x + 2, point_index_y + 1, point_index_z),
                        idx(point_index_x + 1, point_index_y + 2, point_index_z),
                        idx(point_index_x, point_index_y + 1, point_index_z),
                        idx(point_index_x, point_index_y, point_index_z + 1),
                        idx(point_index_x + 2, point_index_y, point_index_z + 1),
                        idx(point_index_x + 2, point_index_y + 2, point_index_z + 1),
                        idx(point_index_x, point_index_y + 2, point_index_z + 1),
                        idx(point_index_x + 1, point_index_y, point_index_z + 2),
                        idx(point_index_x + 2, point_index_y + 1, point_index_z + 2),
                        idx(point_index_x + 1, point_index_y + 2, point_index_z + 2),
                        idx(point_index_x, point_index_y + 1, point_index_z + 2),
                        # face centers
                        idx(point_index_x + 1, point_index_y + 1, point_index_z),
                        idx(point_index_x + 1, point_index_y, point_index_z + 1),
                        idx(point_index_x + 2, point_index_y + 1, point_index_z + 1),
                        idx(point_index_x + 1, point_index_y + 2, point_index_z + 1),
                        idx(point_index_x, point_index_y + 1, point_index_z + 1),
                        idx(point_index_x + 1, point_index_y + 1, point_index_z + 2),
                        # volume center
                        idx(point_index_x + 1, point_index_y + 1, point_index_z + 1),
                    ]

                else:
                    raise ValueError("element must be 'hex8' or 'hex27'")

                i_elem = i_x * ny * nz + i_y * nz + i_z
                connectivity[i_elem] = conn

    return points, connectivity


def get_node_sets_from_point_coordinates(points: NDArray) -> dict[str, NDArray]:
    """Return demo node sets for the given points.

    Args:
        points: Point coordinates.

    Returns:
        node_sets: Mapping of node set name to array of node indices.
    """
    min_z = np.min(points[:, 2])
    return {
        "bottom": np.where(np.isclose(points[:, 2], min_z))[0],
        "all": np.arange(len(points), dtype=int),
    }


def create_exodus_mesh(
    exo_path: Path, n_per_dir_hex8: int, n_per_dir_hex27: int
) -> None:
    """Create an exodus mesh with two blocks (hex8 and hex27) and some node sets.

    Args:
        exo_path: Path to the output exodus file.
        n_per_dir_hex8: Factor for the number of hex8 elements in each direction for the first block.
        n_per_dir_hex27: Factor for the number of hex27 elements in each direction for the second block.
    """
    # Get the data for the hex8 block.
    points_1, connectivity_1 = create_structured_hex_mesh(
        nx=n_per_dir_hex8,
        ny=n_per_dir_hex8,
        nz=n_per_dir_hex8,
        element="hex8",
        bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)),
    )
    node_sets_1 = get_node_sets_from_point_coordinates(points_1)
    connectivity_1 += 1  # Exodus connectivity is 1-based

    # Get the data for the hex27 block.
    points_2, connectivity_2 = create_structured_hex_mesh(
        nx=n_per_dir_hex27,
        ny=n_per_dir_hex27,
        nz=n_per_dir_hex27,
        element="hex27",
        bounds=((2.0, 2.5), (0.0, 1.0), (0.0, 2.0)),
    )
    node_sets_2 = get_node_sets_from_point_coordinates(points_2)

    # Exodus connectivity is 1-based and we have to account for the points in the first block.
    connectivity_2 += len(points_1) + 1

    # Since `create_structured_hex_mesh` uses 4C node numbering, we have to reorder the
    # connectivity indices.
    exo_to_vtk = MESH_REPRESENTATION_MAPPINGS["connectivity_mapping_exodus_to_vtk"][27]
    vtk_to_four_c = MESH_REPRESENTATION_MAPPINGS[
        "element_type_and_n_nodes_to_connectivity_mapping_vtk_to_beamme"
    ][(bme.element_type.solid, 27)]
    four_c_to_exo = np.argsort(np.array(exo_to_vtk)[vtk_to_four_c])
    connectivity_2 = connectivity_2[:, four_c_to_exo]

    # Combine the points from both blocks.
    points = np.concatenate([points_1, points_2])

    # Combine the node sets and adjust the node indices.
    node_sets = {
        name: np.concatenate([node_sets_1[name], node_sets_2[name] + len(points_1)]) + 1
        for name in node_sets_1
    }

    # Create the actual exodus file
    with netCDF4.Dataset(exo_path, "w") as nc:
        # Define string length for storing names
        len_string = 256
        nc.createDimension("len_string", len_string)

        # Store nodes
        nc.createDimension("num_nodes", len(points))
        for i, coord in enumerate(["coordx", "coordy", "coordz"]):
            nc.createVariable(coord, "float64", ("num_nodes",))[:] = points[:, i]

        # Store block information
        nc.createDimension("num_block", 2)
        block_id = nc.createVariable("eb_prop1", "int32", ("num_block",))
        block_id[:] = [1, 2]
        block_name = nc.createVariable("eb_names", "S1", ("num_block", "len_string"))
        block_name[0, :] = netCDF4.stringtoarr("hex8", len_string)
        block_name[1, :] = netCDF4.stringtoarr("hex27", len_string)

        # Store connectivity
        for i, (connectivity, elem_type) in enumerate(
            zip([connectivity_1, connectivity_2], ["HEX8", "HEX27"])
        ):
            nc.createDimension(f"num_elements_block_{i + 1}", connectivity.shape[0])
            nc.createDimension(
                f"num_nodes_per_element_block_{i + 1}", connectivity.shape[1]
            )
            conn = nc.createVariable(
                f"connect{i + 1}",
                "int32",
                (f"num_elements_block_{i + 1}", f"num_nodes_per_element_block_{i + 1}"),
            )
            conn[:] = connectivity
            conn.elem_type = elem_type

        # Store node set information
        nc.createDimension("num_node_set", 2)
        node_set_id = nc.createVariable("ns_prop1", "int32", ("num_node_set",))
        node_set_id[:] = [1, 2]
        node_set_name = nc.createVariable(
            "ns_names", "S1", ("num_node_set", "len_string")
        )

        for i, name in enumerate(["bottom", "all"]):
            node_set_name[i, :] = netCDF4.stringtoarr(
                node_set_info_to_string(i + 1, cupy.geometry.surface, name), len_string
            )
            nc.createDimension(f"num_nodes_set_{i + 1}", len(node_sets[name]))
            node_set = nc.createVariable(
                f"node_ns{i + 1}", "int32", (f"num_nodes_set_{i + 1}",)
            )
            node_set[:] = node_sets[name]


def create_exodus_input_file(
    base_path: Path, n_per_dir_hex8: int, n_per_dir_hex27: int
) -> Path:
    """Create a solid input file containing two blocks (hex8 and hex27) in exodus
    format.

    Args:
        base_path: Directory where the input file and the exodus mesh will be created.
        n_per_dir_hex8: Number of hex8 elements in each direction for the first block.
        n_per_dir_hex27: Number of hex27 elements in each direction for the second block.

    Returns:
        Path to the created input file.
    """
    input_file_path = base_path / "solid_with_exo.4C.yaml"
    exo_path = base_path / "solid_with_exo.exo"

    # Create the input file.
    input_file = InputFile()
    input_file.add(
        {
            "STRUCTURE GEOMETRY": {
                "FILE": exo_path.name,
                "SHOW_INFO": "detailed_summary",
                "ELEMENT_BLOCKS": [
                    {"ID": 1, "SOLID": {"HEX8": {"MAT": 1, "KINEM": "nonlinear"}}},
                    {"ID": 2, "SOLID": {"HEX27": {"MAT": 2, "KINEM": "linear"}}},
                ],
            },
            "MATERIALS": [
                {
                    "MAT": 1,
                    "MAT_Struct_StVenantKirchhoff": {"DENS": 1, "NUE": 0.3, "YOUNG": 2},
                },
                {
                    "MAT": 2,
                    "MAT_Struct_StVenantKirchhoff": {"DENS": 2, "NUE": 0.3, "YOUNG": 2},
                },
            ],
        }
    )
    input_file.dump(
        input_file_path,
        mesh_format="yaml",
        validate_sections_only=True,
        add_header_default=False,
        add_header_information=False,
        add_footer_application_script=False,
    )

    # Create the actual exodus mesh file.
    create_exodus_mesh(
        exo_path, n_per_dir_hex8=n_per_dir_hex8, n_per_dir_hex27=n_per_dir_hex27
    )

    return input_file_path
