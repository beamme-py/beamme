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
"""This function allows to reorder the connectivity of solid shell elements
such that the solid shell direction is correctly represented."""

from typing import List as _List

import numpy as _np
import pyvista as _pv

from beamme.core.element import Element as _Element
from beamme.core.element_volume import VolumeElement as _VolumeElement
from beamme.core.element_volume import VolumeHEX8 as _VolumeHEX8
from beamme.utils.nodes import get_nodal_coordinates as _get_nodal_coordinates


def shape_functions_hex8(xi1, xi2, xi3):
    """Return the shape functions for a hex8 element."""
    shape_functions = _np.zeros(8)
    one_over_eight = 0.125
    shape_functions[0] = one_over_eight * (1 - xi1) * (1 - xi2) * (1 - xi3)
    shape_functions[1] = one_over_eight * (1 + xi1) * (1 - xi2) * (1 - xi3)
    shape_functions[2] = one_over_eight * (1 + xi1) * (1 + xi2) * (1 - xi3)
    shape_functions[3] = one_over_eight * (1 - xi1) * (1 + xi2) * (1 - xi3)
    shape_functions[4] = one_over_eight * (1 - xi1) * (1 - xi2) * (1 + xi3)
    shape_functions[5] = one_over_eight * (1 + xi1) * (1 - xi2) * (1 + xi3)
    shape_functions[6] = one_over_eight * (1 + xi1) * (1 + xi2) * (1 + xi3)
    shape_functions[7] = one_over_eight * (1 - xi1) * (1 + xi2) * (1 + xi3)
    return shape_functions


def shape_functions_derivative_hex8(xi1, xi2, xi3):
    """Return the derivative of the shape functions for a hex8 element."""
    derivatives = _np.zeros((3, 8))
    one_over_eight = 0.125

    # Derivatives with respect to xi1
    derivatives[0, 0] = -one_over_eight * (1 - xi2) * (1 - xi3)
    derivatives[0, 1] = one_over_eight * (1 - xi2) * (1 - xi3)
    derivatives[0, 2] = one_over_eight * (1 + xi2) * (1 - xi3)
    derivatives[0, 3] = -one_over_eight * (1 + xi2) * (1 - xi3)
    derivatives[0, 4] = -one_over_eight * (1 - xi2) * (1 + xi3)
    derivatives[0, 5] = one_over_eight * (1 - xi2) * (1 + xi3)
    derivatives[0, 6] = one_over_eight * (1 + xi2) * (1 + xi3)
    derivatives[0, 7] = -one_over_eight * (1 + xi2) * (1 + xi3)

    # Derivatives with respect to xi2
    derivatives[1, 0] = -one_over_eight * (1 - xi1) * (1 - xi3)
    derivatives[1, 1] = -one_over_eight * (1 + xi1) * (1 - xi3)
    derivatives[1, 2] = one_over_eight * (1 + xi1) * (1 - xi3)
    derivatives[1, 3] = one_over_eight * (1 - xi1) * (1 - xi3)
    derivatives[1, 4] = -one_over_eight * (1 - xi1) * (1 + xi3)
    derivatives[1, 5] = -one_over_eight * (1 + xi1) * (1 + xi3)
    derivatives[1, 6] = one_over_eight * (1 + xi1) * (1 + xi3)
    derivatives[1, 7] = one_over_eight * (1 - xi1) * (1 + xi3)

    # Derivatives with respect to xi3
    derivatives[2, 0] = -one_over_eight * (1 - xi1) * (1 - xi2)
    derivatives[2, 1] = -one_over_eight * (1 + xi1) * (1 - xi2)
    derivatives[2, 2] = -one_over_eight * (1 + xi1) * (1 + xi2)
    derivatives[2, 3] = -one_over_eight * (1 - xi1) * (1 + xi2)
    derivatives[2, 4] = one_over_eight * (1 - xi1) * (1 - xi2)
    derivatives[2, 5] = one_over_eight * (1 + xi1) * (1 - xi2)
    derivatives[2, 6] = one_over_eight * (1 + xi1) * (1 + xi2)
    derivatives[2, 7] = one_over_eight * (1 - xi1) * (1 + xi2)

    return derivatives


def get_hex8_element_center_and_jacobian_mapping(element):
    """Return the center of a hex8 element and the Jacobian mapping for that
    point."""

    nodal_coordinates = _get_nodal_coordinates(element.nodes)
    if not len(nodal_coordinates) == 8:
        raise ValueError(f"Expected 8 nodes, got {len(nodal_coordinates)}")

    N = shape_functions_hex8(0, 0, 0)
    dN = shape_functions_derivative_hex8(0, 0, 0)

    reference_position_center = _np.dot(N, nodal_coordinates)
    jacobian_center = _np.dot(dN, nodal_coordinates)

    return reference_position_center, jacobian_center


def get_reordering_index_thickness(jacobian, *, identify_threshold=None):
    """Return the reordering index from the Jacobian such that the thinnest
    direction is the 3rd parameter direction.

    Additionally it is checked, that the thinnest direction is at least
    identify_threshold times smaller than the next thinnest, to avoid
    wrongly detected directions.
    """

    # The direction with the smallest parameter derivative is the thickness direction
    parameter_derivative_norms = [
        _np.linalg.norm(parameter_direction) for parameter_direction in jacobian
    ]
    thickness_direction = _np.argmin(parameter_derivative_norms)

    if identify_threshold is not None:
        # Check that the minimal parameter direction is at least a given factor
        # smaller than the other directions. This helps to identify cases where
        # it is unlikely that a unique direction is found.
        min_norm = parameter_derivative_norms[thickness_direction]
        relative_difference = [
            parameter_derivative_norms[i] / min_norm
            for i in range(3)
            if not i == thickness_direction
        ]
        if _np.min(relative_difference) < 1.5:
            raise ValueError("Could not uniquely identify the thickness direction.")

    return thickness_direction


def get_reordering_index_director_projection(
    jacobian, director, *, identify_threshold=None
):
    """Return the reordering index from the Jacobian such that the thickness
    direction is the one that has the largest dot product with the given
    director."""

    projections = []
    for parameter_director in jacobian:
        parameter_director = parameter_director / _np.linalg.norm(parameter_director)
        projections.append(_np.abs(_np.dot(director, parameter_director)))
    thickness_direction = _np.argmax(projections)

    if identify_threshold is not None:
        # Check that the maximal dot product is at least a given factor larger than
        # the other projections. This helps to identify cases where it is unlikely
        # that a unique direction is found.
        max_norm = projections[thickness_direction]
        relative_difference = [
            projections[i] / max_norm for i in range(3) if not i == thickness_direction
        ]
        if _np.max(relative_difference) > 1.0 / identify_threshold:
            raise ValueError("Could not uniquely identify the thickness direction.")

    return thickness_direction


def set_solid_shell_thickness_direction(
    elements: _List[_Element],
    *,
    selection_type="thickness",
    director=None,
    director_function=None,
    identify_threshold=2.0,
):
    """Set the solid shell directions for all solid shell elements in the
    element list.

    Args:
    ----
    elements:
        A list containing all elements that should be checked
    selection_type:
        The type of algorithm that shall be used to select the thickness direction
            "thickness":
                The "smallest" dimension of the element will be set to the
                thickness direction
            "projection_director":
                The parameter director that aligns most with a given director
                will be set as the thickness direction
            "projection_director_function":
                The parameter director that aligns most with a director obtained
                by a given function of the element centroid coordinates will be
                set as the thickness direction
    identify_threshold: float/None
        To ensure that the found directions are well-defined, i.e., that not multiple
        directions are almost equally suited to the thickness direction.
    """

    if len(elements) == 0:
        raise ValueError("Expected a non empty element list")

    for element in elements:
        is_hex8_solid_shell = (
            getattr(type(element), "data", {})
            .get("SOLID", {})
            .get("HEX8", {})
            .get("TECH", "")
            == "shell_eas_ans"
        )
        if is_hex8_solid_shell:
            # Get the element center and the Jacobian at the center
            (
                reference_position_center,
                jacobian_center,
            ) = get_hex8_element_center_and_jacobian_mapping(element)

            # Depending on the chosen method, get the thickness direction
            if selection_type == "thickness":
                thickness_direction = get_reordering_index_thickness(
                    jacobian_center, identify_threshold=identify_threshold
                )
            elif selection_type == "projection_director":
                thickness_direction = get_reordering_index_director_projection(
                    jacobian_center, director, identify_threshold=identify_threshold
                )
            elif selection_type == "projection_director_function":
                director = director_function(reference_position_center)
                thickness_direction = get_reordering_index_director_projection(
                    jacobian_center, director, identify_threshold=identify_threshold
                )
            else:
                raise ValueError(
                    f'Got unexpected selection_type of value "{selection_type}"'
                )

            n_apply_mapping = 0
            if thickness_direction == 2:
                # We already have the orientation we want
                continue
            elif thickness_direction == 1:
                # We need to apply the connectivity mapping once, i.e., the 2nd parameter
                # direction has to become the 3rd
                n_apply_mapping = 1
            elif thickness_direction == 0:
                # We need to apply the connectivity mapping twice, i.e., the 1nd parameter
                # direction has to become the 3rd
                n_apply_mapping = 2

            # This permutes the parameter coordinate the following way:
            # [xi,eta,zeta]->[zeta,xi,eta]
            mapping = [2, 6, 7, 3, 1, 5, 4, 0]
            for _ in range(n_apply_mapping):
                element.nodes = [element.nodes[local_index] for local_index in mapping]


def get_visualization_third_parameter_direction_hex8(mesh):
    """Return a pyvista mesh with cell data for the third parameter direction
    for hex8 elements."""

    vtk_solid = mesh.get_vtk_representation()[1].grid
    pv_solid = _pv.UnstructuredGrid(vtk_solid)

    cell_thickness_direction = []
    for element in mesh.elements:
        if isinstance(element, _VolumeHEX8):
            _, jacobian_center = get_hex8_element_center_and_jacobian_mapping(element)
            cell_thickness_direction.append(jacobian_center[2])
        elif isinstance(element, _VolumeElement):
            cell_thickness_direction.append([0, 0, 0])

    if not len(cell_thickness_direction) == pv_solid.number_of_cells:
        raise ValueError(
            "Expected the same number of cells from the mesh and from the "
            f"pyvista object. Got {len(cell_thickness_direction)} form the "
            f"mesh and {pv_solid.number_of_cells} form pyvista"
        )

    pv_solid.cell_data["thickness_direction"] = cell_thickness_direction
    return pv_solid


def visualize_third_parameter_direction_hex8(mesh):
    """Visualize the third parameter direction for hex8 elements.

    This can be used to check the correct definition of the shell
    thickness for solid shell elements.
    """

    grid = get_visualization_third_parameter_direction_hex8(mesh)
    grid = grid.clean()
    cell_centers = grid.cell_centers()
    thickness_direction = cell_centers.glyph(
        orient="thickness_direction", scale="thickness_direction", factor=5
    )

    plotter = _pv.Plotter()
    plotter.renderer.add_axes()
    plotter.add_mesh(grid, color="white", show_edges=True, opacity=0.5)
    plotter.add_mesh(thickness_direction, color="red")
    plotter.show()
