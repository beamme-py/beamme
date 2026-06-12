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
    material = MaterialReissner(
        radius=0.25 / np.max([n_x, n_y, n_z]), youngs_modulus=1.0
    )

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


@pytest.fixture(scope="module")
def medium_solid_block(shared_tmp_path, evaluate_execution_time):
    """Provide a solid mesh from cubit.

    The version of Cubit we use in testing only allows for 50,000, so we
    create a mesh with exactly that.
    """
    cubit = CubitPy()
    input_file_path = shared_tmp_path / "performance_testing_solid_half.4C.yaml"
    evaluate_execution_time(
        "CubitPy: Create half solid block",
        create_solid_block,
        kwargs={
            "cubit": cubit,
            "file_path": input_file_path,
            "nx": 100,
            "ny": 50,
            "nz": 10,
        },
        expected_time=3.8,
    )
    return input_file_path


@pytest.mark.cubitpy
@pytest.mark.performance
def test_performance_beamme_cubitpy_create_solid(medium_solid_block):
    """Test the performance of creating a solid block using CubitPy.

    The test is run in the fixture, so we don't need to do anything
    here.
    """
    pass


@pytest.fixture(scope="module")
def large_solid_block(evaluate_execution_time, medium_solid_block, shared_tmp_path):
    """The version of Cubit we use in testing only allows for 50,000 elements,
    our goal is 100,000 so we double the block with 50,000 elements here."""

    def double_block():
        """Load the block with 50,000 elements and double the mesh."""
        input_file, mesh = import_four_c_model(
            medium_solid_block, convert_input_to_mesh=True
        )
        mesh_translated = mesh.copy()
        mesh_translated.translate([1, 0, 0])
        input_file.add(mesh)
        input_file.add(mesh_translated)
        input_file_path = shared_tmp_path / "performance_testing_solid.4C.yaml"
        input_file.dump(input_file_path, validate=False)
        return input_file_path

    return evaluate_execution_time(
        "BeamMe: Load and double solid element block from cubit",
        double_block,
        expected_time=10.5,
    )


@pytest.mark.cubitpy
@pytest.mark.performance
def test_performance_beamme_double_solid_block(large_solid_block):
    """Test the performance of doubling a solid block from an input file.

    The test is run in the fixture, so we don't need to do anything
    here.
    """
    pass


@pytest.mark.cubitpy
@pytest.mark.parametrize(
    ("log_name", "full_import", "expected_time"),
    [
        ("BeamMe: Load solid mesh (no full import)", False, 2.5),
        ("BeamMe: Load solid mesh (full import)", True, 3.5),
    ],
)
@pytest.mark.performance
def test_performance_beamme_load_solid(
    evaluate_execution_time, large_solid_block, log_name, full_import, expected_time
):
    """Test the performance of loading a solid mesh."""

    evaluate_execution_time(
        log_name,
        import_four_c_model,
        kwargs={
            "input_file_path": large_solid_block,
            "convert_input_to_mesh": full_import,
        },
        expected_time=expected_time,
    )


@pytest.fixture(scope="module")
def large_beam_mesh(evaluate_execution_time):
    """Provide a large beam mesh."""

    return evaluate_execution_time(
        "BeamMe: Create large beam mesh",
        create_beam_mesh,
        kwargs={
            "n_x": 40,
            "n_y": 40,
            "n_z": 10,
            "n_el": 2,
        },
        expected_time=2.5,
    )


@pytest.mark.performance
def test_performance_beamme_create_beams(large_beam_mesh):
    """Test the performance of creating a large beam mesh.

    The test is run in the fixture, so we don't need to do anything
    here.
    """
    pass


@pytest.mark.performance
def test_performance_beamme_copy_beams(large_beam_mesh, evaluate_execution_time):
    """Test the performance of copying a large beam mesh."""

    evaluate_execution_time(
        "BeamMe: Copy large beam mesh",
        large_beam_mesh.copy,
        kwargs={},
        expected_time=7.0,
    )


@pytest.mark.performance
def test_performance_beamme_add_beams_to_mesh(large_beam_mesh, evaluate_execution_time):
    """Test the performance of adding a large beam mesh to another."""

    # To avoid modifying the original mesh, we make a copy and add to that.
    large_beam_mesh_copy = large_beam_mesh.copy()
    evaluate_execution_time(
        "BeamMe: Add large beam mesh to another",
        large_beam_mesh_copy.add,
        args=[large_beam_mesh],
        expected_time=0.15,
    )


@pytest.mark.performance
def test_performance_beamme_rotate(large_beam_mesh, evaluate_execution_time):
    """Test the performance of rotating a large beam mesh."""

    # To avoid an expensive copy, we rotate the mesh back at the end of the test.
    rotation = Rotation([1, 1, 0], np.pi / 3)
    evaluate_execution_time(
        "BeamMe: Rotate large beam mesh",
        large_beam_mesh.rotate,
        kwargs={"rotation": rotation},
        expected_time=0.5,
    )
    large_beam_mesh.rotate(rotation.inv())


@pytest.mark.performance
def test_performance_beamme_translate(large_beam_mesh, evaluate_execution_time):
    """Test the performance of translating a large beam mesh."""

    # To avoid an expensive copy, we move the mesh back at the end of the test.
    distance = np.array([0.5, 0, 0])
    evaluate_execution_time(
        "BeamMe: Translate large beam mesh",
        large_beam_mesh.translate,
        kwargs={"vector": distance},
        expected_time=0.25,
    )
    large_beam_mesh.translate(-distance)


@pytest.mark.performance
def test_performance_beamme_reflect(large_beam_mesh, evaluate_execution_time):
    """Test the performance of reflecting a large beam mesh."""

    # To avoid modifying the original mesh, we make a copy and reflect that.
    large_beam_mesh_copy = large_beam_mesh.copy()
    evaluate_execution_time(
        "BeamMe: Reflect large beam mesh",
        large_beam_mesh_copy.reflect,
        kwargs={"normal_vector": [0.5, 0.4, 0.1]},
        expected_time=0.5,
    )


@pytest.mark.performance
def test_performance_beamme_wrap_around_cylinder(
    large_beam_mesh, evaluate_execution_time
):
    """Test the performance of wrapping a large beam mesh around a cylinder."""

    # To avoid modifying the original mesh, we make a copy and wrap that.
    large_beam_mesh_copy = large_beam_mesh.copy()
    evaluate_execution_time(
        "BeamMe: Wrap large beam mesh around cylinder",
        large_beam_mesh_copy.wrap_around_cylinder,
        kwargs={"radius": 1.0},
        expected_time=1.75,
    )


@pytest.mark.performance
def test_performance_beamme_wrap_around_cylinder_without_check(
    large_beam_mesh, evaluate_execution_time
):
    """Test the performance of wrapping a large beam mesh around a cylinder
    without checking for advanced warnings."""

    # To avoid modifying the original mesh, we make a copy and wrap that.
    large_beam_mesh_copy = large_beam_mesh.copy()
    evaluate_execution_time(
        "BeamMe: Wrap large beam mesh around cylinder without check",
        large_beam_mesh_copy.wrap_around_cylinder,
        kwargs={"radius": 1.0, "advanced_warning": False},
        expected_time=0.5,
    )


@pytest.mark.performance
def test_performance_beamme_find_close_nodes(large_beam_mesh, evaluate_execution_time):
    """Test the performance of finding close nodes in a large beam mesh."""

    evaluate_execution_time(
        "BeamMe: Find close nodes in large beam mesh",
        find_close_nodes,
        kwargs={"nodes": large_beam_mesh.nodes},
        expected_time=0.4,
    )


@pytest.mark.performance
def test_performance_beamme_couple_nodes(large_beam_mesh, evaluate_execution_time):
    """Test the performance of coupling nodes in a large beam mesh.

    We add the mesh to itself, to have matching nodes to replace.
    """

    # To avoid modifying the original mesh, we make a copies and add them to each
    # other.
    large_beam_mesh_copy_1 = large_beam_mesh.copy()
    large_beam_mesh_copy_2 = large_beam_mesh.copy()
    large_beam_mesh_copy_1.add(large_beam_mesh_copy_2)
    evaluate_execution_time(
        "BeamMe: Couple nodes in large beam mesh",
        large_beam_mesh_copy_1.couple_nodes,
        kwargs={"reuse_matching_nodes": True},
        expected_time=7.0,
    )


@pytest.fixture(scope="module")
def large_beam_input_file(large_beam_mesh, evaluate_execution_time):
    """Provide a large input file containing a beam mesh."""

    input_file = InputFile()
    evaluate_execution_time(
        "BeamMe: Add large beam mesh to input file",
        input_file.add,
        kwargs={"object_to_add": large_beam_mesh},
        expected_time=0.6,
    )
    return input_file


@pytest.mark.performance
def test_performance_beamme_add_mesh_to_input_file(large_beam_input_file):
    """Test the performance of adding a mesh to an input file.

    The test is run in the fixture, so we don't need to do anything
    here.
    """
    pass


@pytest.mark.performance
@pytest.mark.parametrize(
    ("log_name", "mesh_format", "expected_time"),
    [
        ("BeamMe: Dump input file with large beam mesh (yaml)", "yaml", 15.0),
        ("BeamMe: Dump input file with large beam mesh (vtu)", "vtu", 0.65),
    ],
)
def test_performance_beamme_dump_input_file(
    log_name,
    mesh_format,
    expected_time,
    large_beam_input_file,
    evaluate_execution_time,
    tmp_path,
):
    """Test the performance of dumping an input file with a large beam mesh."""

    evaluate_execution_time(
        log_name,
        large_beam_input_file.dump,
        kwargs={
            "input_file_path": tmp_path / "performance_testing_beam.4C.yaml",
            "validate_sections_only": True,
            "mesh_format": mesh_format,
        },
        expected_time=expected_time,
    )


@pytest.mark.performance
def test_performance_beamme_write_vtu(
    large_beam_mesh, evaluate_execution_time, tmp_path
):
    """Test the performance of writing a beam mesh to VTU format."""

    evaluate_execution_time(
        "BeamMe: Write beam mesh to VTU",
        large_beam_mesh.write_vtu,
        args=[tmp_path / "performance_testing_beam.vtu"],
        expected_time=2.0,
    )
