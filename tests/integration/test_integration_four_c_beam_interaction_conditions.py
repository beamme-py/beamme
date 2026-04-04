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
"""Integration tests for the beam interaction condition functionality."""

import pytest

from beamme.core.conf import bme
from beamme.core.mesh import Mesh
from beamme.four_c.beam_interaction_conditions import add_beam_interaction_condition
from beamme.four_c.element_beam import Beam3rHerm2Line3
from beamme.four_c.model_importer import import_four_c_model
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line


def test_integration_four_c_beam_interaction_conditions_beam_to_beam_contact(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test the beam-to-beam contact boundary conditions."""

    # Create the mesh.
    mesh = Mesh()

    # Create Material.
    mat = get_default_test_beam_material(material_type="reissner")

    # Create a beam in x-axis.
    beam_x = create_beam_mesh_line(
        mesh,
        Beam3rHerm2Line3,
        mat,
        [0, 0, 0],
        [1, 0, 0],
        n_el=2,
    )

    # Create a second beam in y-axis.
    beam_y = create_beam_mesh_line(
        mesh,
        Beam3rHerm2Line3,
        mat,
        [0, 0, 0.5],
        [1, 0, 0.5],
        n_el=2,
    )

    # Add the beam-to-beam contact condition.
    add_beam_interaction_condition(
        mesh, beam_x["line"], beam_y["line"], bme.bc.beam_to_beam_contact
    )

    # Compare with the reference solution.
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_integration_four_c_beam_interaction_conditions_beam_to_solid(
    get_default_test_beam_material,
    get_corresponding_reference_file_path,
    assert_results_close,
):
    """Test that the automatic ID creation for beam-to-solid conditions
    works."""

    # Load a solid
    _, mesh = import_four_c_model(
        input_file_path=get_corresponding_reference_file_path(
            reference_file_base_name="test_other_create_cubit_input_files_block"
        ),
        convert_input_to_mesh=True,
    )

    # The yaml file already contains the beam-to-solid boundary conditions
    # for the solid. We don't need them in this test case, as we want to
    # create them again. Thus, we have to delete them here.
    mesh.boundary_conditions[
        (bme.bc.beam_to_solid_volume_meshtying, bme.geo.volume)
    ].clear()
    mesh.boundary_conditions[
        (bme.bc.beam_to_solid_surface_meshtying, bme.geo.surface)
    ].clear()

    # Get the geometry set objects representing the geometry from the cubit
    # file.
    surface_set = mesh.geometry_sets[bme.geo.surface][0]
    volume_set = mesh.geometry_sets[bme.geo.volume][0]

    # Add the beam
    material = get_default_test_beam_material(material_type="reissner")
    beam_set_1 = create_beam_mesh_line(
        mesh, Beam3rHerm2Line3, material, [0, 0, 0], [0, 0, 1], n_el=1
    )
    beam_set_2 = create_beam_mesh_line(
        mesh, Beam3rHerm2Line3, material, [0, 1, 0], [0, 1, 1], n_el=2
    )
    add_beam_interaction_condition(
        mesh,
        volume_set,
        beam_set_1["line"],
        bme.bc.beam_to_solid_volume_meshtying,
    )
    add_beam_interaction_condition(
        mesh,
        volume_set,
        beam_set_2["line"],
        bme.bc.beam_to_solid_volume_meshtying,
    )
    add_beam_interaction_condition(
        mesh,
        surface_set,
        beam_set_2["line"],
        bme.bc.beam_to_solid_surface_meshtying,
    )
    add_beam_interaction_condition(
        mesh,
        surface_set,
        beam_set_1["line"],
        bme.bc.beam_to_solid_surface_meshtying,
    )

    assert_results_close(get_corresponding_reference_file_path(), mesh)

    a, _, _, _ = mesh.get_mesh_representation()
    a.save("mesh.vtu")

    # If we try to add this the IDs won't match, because the next volume ID for
    # beam-to-surface coupling should be 0 (this one does not make sense, but
    # this is checked in a later test) and the next line ID for beam-to-surface
    # coupling is 2 (there are already two of these conditions).
    with pytest.raises(ValueError):
        add_beam_interaction_condition(
            mesh,
            volume_set,
            beam_set_1["line"],
            bme.bc.beam_to_solid_surface_meshtying,
        )

    # If we add a wrong geometries to the mesh, the creation of the input file
    # should fail, because there is no beam-to-surface contact section that
    # contains a volume set.
    with pytest.raises(KeyError):
        add_beam_interaction_condition(
            mesh,
            volume_set,
            beam_set_1["line"],
            bme.bc.beam_to_solid_surface_contact,
        )
        assert_results_close(get_corresponding_reference_file_path(), mesh)
