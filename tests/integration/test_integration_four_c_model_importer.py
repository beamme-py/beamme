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
    _extract_mesh_sections,
    import_cubitpy_model,
    import_four_c_model,
)
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line
from tests.create_test_models import create_tube_cubit


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

    # Extract the mesh again. Only one material should be present in the extracted mesh.
    _, mesh_extracted = _extract_mesh_sections(input_file)
    assert len(mesh_extracted.materials) == 1

    # Check that the imported mesh can be added to a mesh that already contains a
    # material. This tests that the nested materials are correctly relinked during
    # the import.
    mesh_with_other_material = Mesh()
    mesh_with_other_material.add(
        get_default_test_solid_material(material_type="st_venant_kirchhoff")
    )
    input_file = InputFile()
    input_file.add(mesh_with_other_material)
    input_file.add(mesh_extracted)

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
            "Material ID 3 not in material_id_map_all (available IDs: [1, 2])."
        ),
    ):
        _extract_mesh_sections(input_file)


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
