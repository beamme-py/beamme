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
"""This file has functions to create a stent according to Auricchio 2012."""

import numpy as _np

from beamme.core.geometry_set import GeometryName as _GeometryName
from beamme.core.geometry_set import GeometrySet as _GeometrySet
from beamme.core.mesh import Mesh as _Mesh
from beamme.core.rotation import Rotation as _Rotation
from beamme.mesh_creation_functions.beam_arc import (
    create_beam_mesh_arc_segment_via_rotation as _create_beam_mesh_arc_segment_via_rotation,
)
from beamme.mesh_creation_functions.beam_line import (
    create_beam_mesh_line as _create_beam_mesh_line,
)
from beamme.utils.nodes import get_min_max_nodes as _get_min_max_nodes


def create_stent_cell(
    beam_class,
    material,
    width,
    height,
    fac_bottom=0.6,
    fac_neck=0.55,
    fac_radius=0.36,
    alpha=0.46 * _np.pi,
    S1=True,
    S2=True,
    S3=True,
    n_el=1,
):
    """Create a cell of the stent.

    This cell is on the x-y plane.

    Args
    ----
    beam_class: Beam
        Class that will be used to create the beam elements.
    material: Material
        Material for the beam.
    width: float
        Width of the total cell.
    height: float
        Height of the total cell.
    fac_bottom: the ratio of the bottom's width to the cell's width
    fac_neck: the ratio of the neck's width to the cell's width
    fac_radius: the ratio of the S1's radius to the cell's width
    alpha: radiant
        The angle between the lines and horizontal line
    n_el: int
        Number of elements per beam line.
    S1, S2, S3: bool
        This check weather the curve S1, S2 or S3 will be created.
        If the cell is on bottom of the stent flat S1 and S2 won't
        be created. If the cell is on top of the flat S1 and S3
        won't be created

    ( these variables are described in a file )

    Return
    ----
    mesh: Mesh
        A mesh with this structure
    """
    mesh = _Mesh()

    def add_line(pointa, pointb, n_el_line):
        """Shortcut to add line."""
        return _create_beam_mesh_line(
            mesh, beam_class, material, pointa, pointb, n_el=n_el_line
        )

    def add_segment(center, axis_rotation, radius, angle, n_el_segment):
        """Shortcut to add arc segment."""
        return _create_beam_mesh_arc_segment_via_rotation(
            mesh,
            beam_class,
            material,
            center,
            axis_rotation,
            radius,
            angle,
            n_el=n_el_segment,
        )

    neck_width = width * fac_neck
    bottom_width = width * fac_bottom
    top_width = 2 * neck_width - bottom_width
    radius = width * fac_radius

    if S1:
        neck_point = _np.array([-neck_width, height * 0.5, 0])
        d = (height * 0.5 / _np.tan(alpha) + bottom_width - neck_width) / _np.sin(alpha)
        CM = _np.array([-_np.sin(alpha), -_np.cos(alpha), 0]) * (d - radius)
        MO = _np.array([_np.cos(alpha), -_np.sin(alpha), 0]) * _np.sqrt(
            radius**2 - (d - radius) ** 2
        )
        S1_angle = _np.pi / 2 + _np.arcsin((d - radius) / radius)
        S1_center1 = CM + MO + neck_point
        S1_axis_rotation1 = _Rotation([0, 0, 1], 2 * _np.pi - S1_angle - alpha)
        add_segment(S1_center1, S1_axis_rotation1, radius, S1_angle, n_el)
        add_line([-bottom_width, 0, 0], mesh.nodes[-1].coordinates, 2 * n_el)

        S1_center2 = 2 * neck_point - S1_center1
        S1_axis_rotation2 = _Rotation([0, 0, 1], _np.pi - alpha - S1_angle)

        add_segment(S1_center2, S1_axis_rotation2, radius, S1_angle, n_el)
        add_line(
            mesh.nodes[-1].coordinates, 2 * neck_point - [-bottom_width, 0, 0], 2 * n_el
        )

    if S3:
        S3_radius = (width - bottom_width + height * 0.5 / _np.tan(alpha)) * _np.tan(
            alpha / 2
        )
        S3_center = [-width, height * 0.5, 0] + S3_radius * _np.array([0, 1, 0])
        S3_axis_rotation = _Rotation()
        S3_angle = _np.pi - alpha
        add_segment(S3_center, S3_axis_rotation, S3_radius, S3_angle, n_el)
        add_line(mesh.nodes[-1].coordinates, [-bottom_width, height, 0], 2 * n_el)

    if S2:
        S2_radius = (height * 0.5 / _np.tan(alpha) + top_width) * _np.tan(alpha * 0.5)
        S2_center = [0, height * 0.5, 0] - S2_radius * _np.array([0, 1, 0])
        S2_angle = _np.pi - alpha
        S2_axis_rotation = _Rotation([0, 0, 1], _np.pi)
        add_segment(S2_center, S2_axis_rotation, S2_radius, S2_angle, 2 * n_el)
        add_line(mesh.nodes[-1].coordinates, [-top_width, 0, 0], 2 * n_el)

    mesh.translate([width, -height / 2, 0])
    return mesh


def create_stent_column(
    beam_class, material, width, height, n_height, n_el=1, **kwargs
):
    """Create a column of completed cells.

    A completed cell consists of one cell, that is created with the create cell
    function and it's reflection.

    Args
    ----
    beam_class: Beam
        Class that will be used to create the beam elements.
    material: Material
        Material for the beam.
    width: float
        Width of the total cell.
    height: float
        Height of the total cell.
    n_height: int
        The number of cells in the column
    n_el: int
        Number of elements per beam line.
    ( these variables are described in a file )

    Return
    ----
    mesh: Mesh
        A mesh with this structure.
    """
    mesh_column = _Mesh()
    for i in range(n_height):
        S1 = True
        S2 = True
        S3 = True
        if i == 0:
            S1 = False
            S2 = False
        if i == 1:
            S2 = False
        if i == n_height - 1:
            S1 = False
            S3 = False

        if i == n_height - 2:
            S3 = False
        unit_cell = create_stent_cell(
            beam_class,
            material,
            width,
            height,
            S1=S1,
            S2=S2,
            S3=S3,
            n_el=n_el,
            **kwargs,
        )
        unit_cell.translate([0, i * height, 0])
        mesh_column.add(unit_cell)

    column_copy = mesh_column.copy()
    column_copy.reflect(normal_vector=[1, 0, 0], origin=[width, 0, 0])
    mesh_column.add(column_copy)

    return mesh_column


def create_beam_mesh_stent_flat(
    beam_class, material, width_flat, height_flat, n_height, n_column, n_el=1, **kwargs
):
    """Create a flat stent structure on the x-y plane.

    Args
    ----
    beam_class: Beam
        Class that will be used to create the beam elements.
    material: Material
        Material for the beam.
    width_flat: float
        The width of the flat structure.
    height_flat: float
        The height of the flat structure.
    n_height: int
        The number of cells in y direction.
    n_column: int
        The number of columns in x direction.
    n_el: int
        Number of elements per beam line.

    Return
    ----
    mesh: Mesh
        A mesh with this structure
    """
    mesh_flat = _Mesh()
    width = width_flat / n_column / 2
    height = height_flat / n_height
    column_mesh = create_stent_column(
        beam_class, material, width, height, n_height, n_el=n_el, **kwargs
    )
    for i in range(n_column):
        column_copy = column_mesh.copy()
        column_copy.translate([2 * width * i, 0, 0])
        mesh_flat.add(column_copy)

    for i in range(n_column // 2):
        for j in range(n_height - 1):
            _create_beam_mesh_line(
                mesh_flat,
                beam_class,
                material,
                [4 * i * width, j * height, 0],
                [4 * i * width, (j + 1) * height, 0],
                n_el=2 * n_el,
            )
        _create_beam_mesh_line(
            mesh_flat,
            beam_class,
            material,
            [(4 * i + 2) * width, 0, 0],
            [(4 * i + 2) * width, height, 0],
            n_el=2 * n_el,
        )
    return mesh_flat


def create_beam_mesh_stent(
    mesh,
    beam_class,
    material,
    length,
    diameter,
    n_axis,
    n_circumference,
    **kwargs,
):
    """Create a stent structure around cylinder, The cylinder axis will be the z-axis.

    Args
    ----
    mesh: Mesh
        Mesh that the stent will be added to.
    beam_class: Beam
        Class that will be used to create the beam elements.
    material: Material
        Material for the beam.
    length: float
        The length of this stent.
    diameter: float
        The diameter of the stent's cross section.
    n_axis: int
        Number of cells in axial-direction.
    n_circumference: int
        Number of cells around the diameter.
    ( these variables are described in a file )

    Return
    ----
    return_set: GeometryName
        Set with nodes on the top, bottom boundaries. Those
        sets only contains end nodes of lines, not the middle ones.
        The set 'all' contains all nodes.
    """
    # Only allow even number of columns.
    if n_circumference % 2 == 1:
        raise ValueError("has to be even even number!")

    # Set the Parameter for other functions
    height_flat = length
    width_flat = _np.pi * diameter
    n_height = n_axis
    n_column = n_circumference

    mesh_stent = create_beam_mesh_stent_flat(
        beam_class, material, width_flat, height_flat, n_height, n_column, **kwargs
    )
    mesh_stent.rotate(_Rotation([1, 0, 0], _np.pi / 2))
    mesh_stent.rotate(_Rotation([0, 0, 1], _np.pi / 2))
    mesh_stent.translate([diameter / 2, 0, 0])
    mesh_stent.wrap_around_cylinder()

    # List of nodes from the stent that are candidates for connections.
    stent_nodes = [node for node in mesh_stent.nodes if node.is_end_node]

    # Add connections for the nodes with same positions.
    mesh_stent.couple_nodes(nodes=stent_nodes)

    # Get min and max nodes of the honeycomb.
    min_max_nodes = _get_min_max_nodes(stent_nodes)

    # Return the geometry set.
    return_set = _GeometryName()
    return_set["top"] = min_max_nodes["z_max"]
    return_set["bottom"] = min_max_nodes["z_min"]
    return_set["all"] = _GeometrySet(mesh_stent.elements)

    mesh.add_mesh(mesh_stent)

    return return_set
