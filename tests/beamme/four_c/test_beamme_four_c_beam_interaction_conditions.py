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
"""Unit tests for the beam interaction condition functionality."""

from beamme.core.boundary_condition import BoundaryCondition
from beamme.core.conf import bme
from beamme.core.geometry_set import GeometrySet
from beamme.core.mesh import Mesh
from beamme.four_c.beam_interaction_conditions import add_beam_interaction_condition
from beamme.four_c.element_beam import Beam3rHerm2Line3
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line


def test_beamme_four_c_beam_interaction_conditions_condition_id(
    get_default_test_beam_material,
):
    """Ensure that the contact-boundary conditions ids are estimated correctly."""
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
        [2, 0, 0],
        n_el=3,
    )

    # Create a second beam in y-axis.
    beam_y = create_beam_mesh_line(
        mesh,
        Beam3rHerm2Line3,
        mat,
        [0, 0, 0],
        [0, 2, 0],
        n_el=3,
    )

    # Add two contact node sets.
    id = add_beam_interaction_condition(
        mesh, beam_x["line"], beam_y["line"], bme.bc.beam_to_beam_contact
    )
    assert id == 0

    # Check if we can add the same set twice.
    id = add_beam_interaction_condition(
        mesh, beam_x["line"], beam_x["line"], bme.bc.beam_to_beam_contact
    )
    assert id == 1

    # Add some more functions to ensure that everything works as expected:
    for node in mesh.nodes:
        mesh.add(
            BoundaryCondition(
                GeometrySet(node),
                "",
                bc_type=bme.bc.dirichlet,
            )
        )

    # Add condition with higher id.
    id = add_beam_interaction_condition(
        mesh, beam_x["line"], beam_x["line"], bme.bc.beam_to_beam_contact, id=3
    )
    assert id == 3

    # Check if the id gap is filled automatically.
    id = add_beam_interaction_condition(
        mesh, beam_x["line"], beam_y["line"], bme.bc.beam_to_beam_contact
    )
    assert id == 2
