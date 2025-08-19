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
"""Create a couple of different mesh cases and test the performance."""

import numpy as np
import pytest

from beamme.core.mesh import Mesh
from beamme.core.rotation import Rotation
from beamme.four_c.element_beam import Beam3rHerm2Line3
from beamme.four_c.input_file import InputFile
from beamme.four_c.material import MaterialReissner
from beamme.four_c.model_importer import import_four_c_model
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line
from beamme.utils.environment import cubitpy_is_available
from beamme.utils.nodes import find_close_nodes

if cubitpy_is_available():
    from cubitpy import CubitPy, cupy


@pytest.fixture(scope="module")
def shared_tmp_path(tmp_path_factory):
    """Create a temporary path for shared use in performance tests."""
    return tmp_path_factory.mktemp("performance_tests")


def create_solid_block(cubit, file_path, nx, ny, nz):
    """Create a solid block (1 x 1 x 1) with (nx * ny * nz) elements."""

    # Create brick.
    brick = cubit.brick(1)

    # Set mesh parameters.
    mesh_size = [
        [nx, [2, 4, 6, 8]],
        [ny, [1, 3, 5, 7]],
        [nz, [9, 10, 11, 12]],
    ]
    for [n_el, curves] in mesh_size:
        for i in curves:
            cubit.set_line_interval(brick.curves()[i - 1], n_el)
    brick.volumes()[0].mesh()

    # Add block and sets.
    cubit.add_element_type(
        brick.volumes()[0],
        cupy.element_type.hex8,
        name="brick",
        material={"MAT": 1},
        bc_description={"KINEM": "nonlinear"},
    )
    counter = 0
    for item in brick.vertices():
        cubit.add_node_set(
            item,
            name="node_set_" + str(counter),
            bc_type=cupy.bc_type.neumann,
            bc_description={
                "NUMDOF": 3,
                "ONOFF": [1, 1, 1],
                "VAL": [3.0, 3.0, 0],
                "FUNCT": [1, 2, 0],
            },
        )
        counter += 1
    for item in brick.curves():
        cubit.add_node_set(
            item,
            name="node_set_" + str(counter),
            bc_type=cupy.bc_type.dirichlet,
            bc_description={
                "NUMDOF": 3,
                "ONOFF": [1, 1, 1],
                "VAL": [3.0, 3.0, 0],
                "FUNCT": [1, 2, 0],
            },
        )
        counter += 1
    for item in brick.surfaces():
        cubit.add_node_set(
            item,
            name="node_set_" + str(counter),
            bc_type=cupy.bc_type.neumann,
            bc_description={
                "NUMDOF": 3,
                "ONOFF": [1, 1, 1],
                "VAL": [3.0, 3.0, 0],
                "FUNCT": [1, 2, 0],
            },
        )
        counter += 1
    for item in brick.volumes():
        cubit.add_node_set(
            item,
            name="node_set_" + str(counter),
            bc_type=cupy.bc_type.neumann,
            bc_description={
                "NUMDOF": 3,
                "ONOFF": [1, 1, 1],
                "VAL": [3.0, 3.0, 0],
                "FUNCT": [1, 2, 0],
            },
        )
        counter += 1

    # Set the material.
    cubit.fourc_input["MATERIALS"] = [
        {"MAT": 1, "MAT_Struct_StVenantKirchhoff": {"DENS": 1, "NUE": 0.3, "YOUNG": 2}}
    ]

    # Export mesh
    cubit.dump(file_path)


def create_beam_mesh(n_x, n_y, n_z, n_el):
    """Create a beam grid on the domain (1 x 1 x 1) with (nx * ny * nz) "grid
    cells"."""

    mesh = Mesh()
    material = MaterialReissner(radius=0.25 / np.max([n_x, n_y, n_z]))

    for i_x in range(n_x + 1):
        for i_y in range(n_y + 1):
            create_beam_mesh_line(
                mesh,
                Beam3rHerm2Line3,
                material,
                [i_x / n_x, i_y / n_y, 0],
                [i_x / n_x, i_y / n_y, 1],
                n_el=n_z * n_el,
            )
    for i_y in range(n_y + 1):
        for i_z in range(n_z + 1):
            create_beam_mesh_line(
                mesh,
                Beam3rHerm2Line3,
                material,
                [0, i_y / n_y, i_z / n_z],
                [1, i_y / n_y, i_z / n_z],
                n_el=n_x * n_el,
            )
    for i_z in range(n_z + 1):
        for i_x in range(n_x + 1):
            create_beam_mesh_line(
                mesh,
                Beam3rHerm2Line3,
                material,
                [i_x / n_x, 0, i_z / n_z],
                [i_x / n_x, 1, i_z / n_z],
                n_el=n_y * n_el,
            )
    return mesh


@pytest.mark.performance
def test_performance_cubitpy_create_solid(evaluate_execution_time, shared_tmp_path):
    """Test the performance of creating a solid block using CubitPy.

    The version of Cubit we use in testing only allows for 50,000, so we
    create a mesh with exactly that.
    """

    cubit = CubitPy()

    evaluate_execution_time(
        "CubitPy: Create half solid block",
        create_solid_block,
        kwargs={
            "cubit": cubit,
            "file_path": shared_tmp_path / "performance_testing_solid_half.4C.yaml",
            "nx": 100,
            "ny": 50,
            "nz": 10,
        },
        expected_time=2.7,
    )


@pytest.mark.performance
def test_performance_double_solid_block(evaluate_execution_time, shared_tmp_path):
    """The version of Cubit we use in testing only allows for 50,000 elements,
    our goal is 100,000 so we double the block with 50,000 elements here."""

    def double_block():
        """Load the block with 50,000 elements and double the mesh."""
        input_file, mesh = import_four_c_model(
            shared_tmp_path / "performance_testing_solid_half.4C.yaml",
            convert_input_to_mesh=True,
        )
        mesh_translated = mesh.copy()
        mesh_translated.translate([1, 0, 0])
        input_file.add(mesh)
        input_file.add(mesh_translated)
        input_file.dump(
            shared_tmp_path / "performance_testing_solid.4C.yaml", validate=False
        )

    evaluate_execution_time(
        "BeamMe: Load and double solid element block from cubit",
        double_block,
        expected_time=10.5,
    )


@pytest.mark.parametrize(
    ("log_name", "full_import", "expected_time"),
    [
        ("BeamMe: Load solid mesh (no full import)", False, 2.5),
        ("BeamMe: Load solid mesh (full import)", True, 3.5),
    ],
)
@pytest.mark.performance
def test_performance_load_solid(
    evaluate_execution_time, shared_tmp_path, log_name, full_import, expected_time
):
    """Test the performance of loading a solid mesh."""

    evaluate_execution_time(
        log_name,
        import_four_c_model,
        kwargs={
            "input_file_path": shared_tmp_path / "performance_testing_solid.4C.yaml",
            "convert_input_to_mesh": full_import,
        },
        expected_time=expected_time,
    )


@pytest.mark.performance
def test_performance_create_beams(evaluate_execution_time, cache_data):
    """Test the performance of creating a large beam mesh."""

    # store mesh in cache for upcoming tests
    cache_data.mesh = evaluate_execution_time(
        "BeamMe: Create large beam mesh",
        create_beam_mesh,
        kwargs={
            "n_x": 40,
            "n_y": 40,
            "n_z": 10,
            "n_el": 2,
        },
        expected_time=5.0,
    )


@pytest.mark.performance
def test_performance_rotate(evaluate_execution_time, cache_data):
    """Test the performance of rotating a large beam mesh."""

    evaluate_execution_time(
        "BeamMe: Rotate large beam mesh",
        cache_data.mesh.rotate,
        kwargs={"rotation": Rotation([1, 1, 0], np.pi / 3)},
        expected_time=0.5,
    )


@pytest.mark.performance
def test_performance_translate(evaluate_execution_time, cache_data):
    """Test the performance of translating a large beam mesh."""

    evaluate_execution_time(
        "BeamMe: Translate large beam mesh",
        cache_data.mesh.translate,
        kwargs={"vector": [0.5, 0, 0]},
        expected_time=0.25,
    )


@pytest.mark.performance
def test_performance_reflect(evaluate_execution_time, cache_data):
    """Test the performance of reflecting a large beam mesh."""

    evaluate_execution_time(
        "BeamMe: Reflect large beam mesh",
        cache_data.mesh.reflect,
        kwargs={"normal_vector": [0.5, 0.4, 0.1]},
        expected_time=0.5,
    )


@pytest.mark.performance
def test_performance_mespy_wrap_around_cylinder(evaluate_execution_time, cache_data):
    """Test the performance of wrapping a large beam mesh around a cylinder."""

    evaluate_execution_time(
        "BeamMe: Wrap large beam mesh around cylinder",
        cache_data.mesh.wrap_around_cylinder,
        kwargs={"radius": 1.0},
        expected_time=1.75,
    )


@pytest.mark.performance
def test_performance_wrap_around_cylinder_without_check(
    evaluate_execution_time, cache_data
):
    """Test the performance of wrapping a large beam mesh around a cylinder
    without checking for advanced warnings."""

    evaluate_execution_time(
        "BeamMe: Wrap large beam mesh around cylinder without check",
        cache_data.mesh.wrap_around_cylinder,
        kwargs={"radius": 1.0, "advanced_warning": False},
        expected_time=0.5,
    )


@pytest.mark.performance
def test_performance_find_close_nodes(evaluate_execution_time, cache_data):
    """Test the performance of finding close nodes in a large beam mesh."""

    evaluate_execution_time(
        "BeamMe: Find close nodes in large beam mesh",
        find_close_nodes,
        kwargs={"nodes": cache_data.mesh.nodes},
        expected_time=0.4,
    )


@pytest.mark.performance
def test_performance_add_mesh_to_input_file(evaluate_execution_time, cache_data):
    """Test the performance of adding a mesh to an input file."""

    input_file = InputFile()

    evaluate_execution_time(
        "BeamMe: Add large beam mesh to input file",
        input_file.add,
        kwargs={"object_to_add": cache_data.mesh},
        expected_time=10.0,
    )

    cache_data.input_file = input_file


@pytest.mark.performance
def test_performance_dump_input_file(
    evaluate_execution_time,
    cache_data,
    tmp_path,
):
    """Test the performance of dumping an input file with a large beam mesh."""

    evaluate_execution_time(
        "BeamMe: Dump input file with large beam mesh",
        cache_data.input_file.dump,
        kwargs={
            "input_file_path": tmp_path / "performance_testing_beam.4C.yaml",
            "validate_sections_only": True,
        },
        expected_time=5.5,
    )


@pytest.mark.performance
def test_performance_write_vtk(evaluate_execution_time, tmp_path, cache_data):
    """Test the performance of writing a beam mesh to VTK format."""

    # use a smaller mesh for testing vtk output performance
    cache_data.mesh = create_beam_mesh(n_x=20, n_y=20, n_z=10, n_el=2)

    evaluate_execution_time(
        "BeamMe: Write beam mesh to VTK",
        cache_data.mesh.write_vtk,
        kwargs={
            "output_name": "performance_testing_beam",
            "output_directory": tmp_path,
            "beam_centerline_visualization_segments": 1,
        },
        expected_time=4.0,
    )


@pytest.mark.performance
def test_performance_write_vtk_smooth(evaluate_execution_time, tmp_path, cache_data):
    """Test the performance of writing a beam mesh to VTK format with more
    segments."""

    evaluate_execution_time(
        "BeamMe: Write beam mesh to VTK with more segments",
        cache_data.mesh.write_vtk,
        kwargs={
            "output_name": "performance_testing_beam",
            "output_directory": tmp_path,
            "beam_centerline_visualization_segments": 5,
        },
        expected_time=7.75,
    )
