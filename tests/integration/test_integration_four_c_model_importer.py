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
"""Integration tests for the 4C model importer of BeamMe."""

import re

import pytest

from beamme.core.mesh import Mesh
from beamme.four_c.element_beam import Beam3rHerm2Line3
from beamme.four_c.input_file import InputFile
from beamme.four_c.model_importer import (
    _extract_materials_from_input_file,
    import_cubitpy_model,
    import_four_c_model,
)
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line
from beamme.utils.environment import cubitpy_is_available
from tests.create_test_models import create_multiple_solid_bricks, create_tube_cubit

if cubitpy_is_available():
    from cubitpy import CubitPy
    from cubitpy.conf import cupy


@pytest.mark.parametrize("full_import", (False, True))
@pytest.mark.cubitpy
def test_integration_four_c_model_importer_import_cubitpy_model(
    full_import, assert_results_close, get_corresponding_reference_file_path
):
    """Check that an import from a cubitpy object works as expected."""

    cubit = create_tube_cubit()
    input_file_cubit, mesh = import_cubitpy_model(
        cubit, convert_input_to_mesh=full_import
    )
    if full_import:
        input_file_cubit.add(mesh)

    assert_results_close(
        get_corresponding_reference_file_path(
            reference_file_base_name="test_other_create_cubit_input_files_tube"
        ),
        input_file_cubit,
    )


@pytest.mark.parametrize("full_import", (False, True))
@pytest.mark.cubitpy
def test_integration_four_c_model_importer_solid_element_types_from_cubitpy(
    full_import, assert_results_close, get_corresponding_reference_file_path
):
    """Check that all supported solid element types are imported correctly from
    a CubitPy instance."""

    cubit = create_multiple_solid_bricks()
    input_file, mesh = import_cubitpy_model(cubit, convert_input_to_mesh=full_import)
    if full_import:
        input_file.add(mesh)

    reference_file = get_corresponding_reference_file_path(
        reference_file_base_name="test_other_create_cubit_input_files_multiple_solid_bricks"
    )
    assert_results_close(reference_file, input_file)


@pytest.mark.parametrize("full_import", (False, True))
def test_integration_four_c_model_importer_solid_element_types_from_input_file(
    full_import, assert_results_close, get_corresponding_reference_file_path
):
    """Check that all supported solid element types are imported correctly from
    a text based input file."""

    reference_file = get_corresponding_reference_file_path(
        reference_file_base_name="test_other_create_cubit_input_files_multiple_solid_bricks"
    )
    input_file, mesh = import_four_c_model(
        input_file_path=reference_file, convert_input_to_mesh=full_import
    )
    if full_import:
        input_file.add(mesh)

    assert_results_close(reference_file, input_file)


@pytest.mark.parametrize("full_import", (False, True))
def test_integration_four_c_model_importer_solid_element_types_from_input_file_with_exodus(
    full_import, assert_results_close, get_corresponding_reference_file_path
):
    """Check that all supported solid element types are imported correctly from
    an input file with an exodus mesh."""

    # We never need to directly reference the exodus file, with `get_corresponding_reference_file_path`
    # so we do a dummy reference here, to avoid a failing reference file check
    # in CI.
    get_corresponding_reference_file_path(extension="exo")

    input_file_path = get_corresponding_reference_file_path()
    input_file, mesh = import_four_c_model(
        input_file_path=input_file_path, convert_input_to_mesh=full_import
    )
    if full_import:
        input_file.add(mesh)

    reference_file_path = get_corresponding_reference_file_path(
        reference_file_base_name="test_other_create_cubit_input_files_multiple_solid_bricks"
    )
    assert_results_close(reference_file_path, input_file)


def test_integration_four_c_model_importer_import_nested_materials(
    get_default_test_solid_material,
    get_corresponding_reference_file_path,
    assert_results_close,
):
    """Check if nested materials are imported correctly."""

    # Create a minimal solid input file.
    mesh = Mesh()
    material = get_default_test_solid_material(material_type="solid_nested")
    mesh.add(material)

    # Add the mesh to an input file, this converts the material to the FourCIPP format.
    input_file = InputFile()
    input_file.add(mesh)

    # Extract the material from the input file. Only one material should be present in the extracted material map.
    material_id_map = _extract_materials_from_input_file(input_file)
    assert len(material_id_map) == 1

    # Check that the extracted material can be added to a mesh that already contains a
    # material. This tests that the nested materials are correctly relinked during
    # the import.
    mesh_with_other_material = Mesh()
    mesh_with_other_material.add(
        get_default_test_solid_material(material_type="st_venant_kirchhoff")
    )
    mesh_with_other_material.add(list(material_id_map.values())[0])
    input_file = InputFile()
    input_file.add(mesh_with_other_material)

    # Compare with reference file.
    assert_results_close(get_corresponding_reference_file_path(), input_file)


def test_integration_four_c_model_importer_import_nested_materials_error():
    """Check that an error is raised when importing nested materials with bad
    IDs."""

    # Create an input file with "bad" material IDs.
    input_file = InputFile()
    input_file["MATERIALS"] = [
        {"MAT": 1, "MAT_ElastHyper": {"NUMMAT": 2, "MATIDS": [2, 3], "DENS": 1.0}},
        {"MAT": 2, "ELAST_CoupSVK": {"YOUNG": 1.0, "NUE": 0.0}},
    ]

    # Try to extract the mesh, this should raise an error.
    with pytest.raises(
        KeyError,
        match=re.escape(
            "Material ID 2 not in material_id_map_all (available IDs: [0, 1])."
        ),
    ):
        _extract_materials_from_input_file(input_file)


@pytest.mark.parametrize("full_import", (False, True))
def test_integration_four_c_model_importer_non_consecutive_geometry_sets(
    full_import,
    get_default_test_beam_material,
    get_corresponding_reference_file_path,
    assert_results_close,
):
    """Test that we can import non-consecutively numbered geometry sets."""

    input_file, mesh = import_four_c_model(
        input_file_path=get_corresponding_reference_file_path(
            additional_identifier="input"
        ),
        convert_input_to_mesh=full_import,
    )

    material = get_default_test_beam_material(material_type="reissner")
    for i in range(3):
        beam_set = create_beam_mesh_line(
            mesh,
            Beam3rHerm2Line3,
            material,
            [i + 3, 0, 0],
            [i + 3, 0, 4],
            n_el=2,
        )
        mesh.add(beam_set)

    input_file.add(mesh)

    assert_results_close(
        get_corresponding_reference_file_path(
            additional_identifier="full_import" if full_import else "dict_import"
        ),
        input_file,
    )


@pytest.mark.cubitpy
@pytest.mark.parametrize("full_import", (False, True))
def test_integration_four_c_model_importer_user_defined_node_set_and_block_ids(
    full_import, get_corresponding_reference_file_path, assert_results_close
):
    """Test that user-defined node set and block IDs work as expected."""

    # Set up Cubit.
    cubit = CubitPy()

    # Initialize geometry
    cubit.cmd("brick x 1 y 1 z 1")
    cubit.cmd("brick x 5e-1 y 5e-1 z 5e-1")
    cubit.cmd("move Volume 2 x 75e-2 y 0 z 0")
    cubit.cmd("volume 1 size {5e-1}")
    cubit.cmd("volume 2 size {2.5e-1}")

    # mesh the two geometries
    cubit.cmd("mesh volume 1")
    cubit.cmd("mesh volume 2")

    # Assign nodesets, required for boundary conditions
    cubit.add_node_set(
        cubit.group(add_value="add surface 6"),
        name="slave",
        bc_type=cupy.bc_type.solid_to_solid_contact,
        bc_description={
            "InterfaceID": 6,
            "Side": "Slave",
        },
    )
    cubit.add_node_set(
        cubit.group(add_value="add surface 10"),
        name="master",
        bc_type=cupy.bc_type.solid_to_solid_contact,
        bc_description={
            "InterfaceID": 10,
            "Side": "Master",
        },
    )
    cubit.add_node_set(
        cubit.group(add_value="add surface 4"),
        name="wall",
        bc_type=cupy.bc_type.dirichlet,
        bc_description={
            "NUMDOF": 3,
            "ONOFF": [4, 0, 0],
            "VAL": [0, 0, 0],
            "FUNCT": [None, None, None],
        },
        node_set_id=17,
    )
    cubit.add_node_set(
        cubit.group(add_value="add surface 12"),
        name="pushing",
        bc_type=cupy.bc_type.dirichlet,
        bc_description={
            "NUMDOF": 3,
            "ONOFF": [12, 0, 0],
            "VAL": [-1.0, 0.0, 0.0],
            "FUNCT": [1, None, None],
        },
    )

    cubit.add_node_set(
        cubit.group(add_value="add curve 1"),
        name="curve_1",
        bc_type=cupy.bc_type.neumann,
        bc_description={
            "NUMDOF": 3,
            "ONOFF": [1, 0, 0],
            "VAL": [1.0, 0.0, 0.0],
            "FUNCT": [None, None, None],
        },
    )
    cubit.add_node_set(
        cubit.group(add_value="add curve 2"),
        name="curve_2",
        bc_type=cupy.bc_type.neumann,
        bc_description={
            "NUMDOF": 3,
            "ONOFF": [2, 0, 0],
            "VAL": [1.0, 0.0, 0.0],
            "FUNCT": [None, None, None],
        },
    )
    cubit.add_node_set(
        cubit.group(add_value="add curve 3"),
        name="curve_3",
        bc_type=cupy.bc_type.neumann,
        bc_description={
            "NUMDOF": 3,
            "ONOFF": [3, 0, 0],
            "VAL": [1.0, 0.0, 0.0],
            "FUNCT": [None, None, None],
        },
        node_set_id=15,
    )

    cubit.add_node_set(cubit.group(add_value="add curve 4"), name="set_without_bc")

    # Add the element types
    cubit.add_element_type(
        cubit.group(add_value="add volume 1"),
        el_type=cupy.element_type.hex8,
        material={
            "MAT": 1,
        },
        bc_description={
            "KINEM": "nonlinear",
        },
        block_id=27,
    )
    cubit.add_element_type(
        cubit.group(add_value="add volume 2"),
        el_type=cupy.element_type.hex8,
        material={
            "MAT": 2,
        },
        bc_description={
            "KINEM": "nonlinear",
        },
        block_id=1,
    )

    # Add the materials
    cubit.fourc_input["MATERIALS"] = [
        {
            "MAT": 1,
            "MAT_Struct_StVenantKirchhoff": {"DENS": 1, "NUE": 0.3, "YOUNG": 2},
        },
        {
            "MAT": 2,
            "MAT_Struct_StVenantKirchhoff": {"DENS": 2, "NUE": 0.3, "YOUNG": 2},
        },
    ]

    # Load into BeamMe
    input_file, mesh = import_cubitpy_model(cubit, convert_input_to_mesh=full_import)
    input_file.add(mesh)

    if full_import:
        named_geometry_sets = mesh.get_named_geometry_sets()
        assert len(named_geometry_sets) == 8
        assert set(
            [
                "curve_1",
                "curve_2",
                "curve_3",
                "master",
                "pushing",
                "set_without_bc",
                "slave",
                "wall",
            ]
        ) == set(named_geometry_sets.keys())

    # Compare with reference file.
    assert_results_close(
        get_corresponding_reference_file_path(
            additional_identifier="full_import" if full_import else "dict_import"
        ),
        input_file,
    )
