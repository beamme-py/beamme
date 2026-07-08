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
"""This file has functions to create NURBS geometries using Geomdl."""

import numpy as _np
from geomdl import NURBS as _NURBS
from geomdl import compatibility as _compat
from geomdl import operations as _operations


def create_nurbs_hollow_cylinder_segment_2d(
    radius_in, radius_out, angle, *, n_ele_u=1, n_ele_v=1
):
    """Creates a patch of a 2 dimensional segment of a hollow cylinder.

    Args
    ----
    radius_in: double
        inner cylinder radius
    radius_out: double
        outer cylinder radius
    angle: double
        angle of the hollow cylinder section (radians)
    n_ele_u: int
        number of elements in the parametric u-direction
    n_ele_v: int
        number of elements in the parametric v-direction


    Return
    ----
    surf: geomdl object
        geomdl object that contains the surface information
    """
    # Check the validity of the input values:
    if radius_in >= radius_out:
        raise ValueError(
            "The external radius should be larger than the internal radius of a hollow cylinder."
        )
    if (angle > _np.pi) or (angle < 0):
        raise ValueError(
            "The following algorithm for creating a hollow cylinder section is only valid for 0 < angle <= pi."
        )

    # Create a NURBS surface instance
    surf = _NURBS.Surface()

    # Set degrees
    surf.degree_u = 2
    surf.degree_v = 2

    # Control points and set them to the surface
    p_size_u = 3
    p_size_v = 3

    # To calculate the internal control point cp_2 we have to calculate
    # the intersection of the tangents of the points cp_1 and cp_3.
    # As point cp_1 is always defined over the x axis, its tangent is simply x = radius
    # The equation of the tangent to the circle at the cp_3 point is:
    # y - cp_3y = -(cp_3x/cp_3y)*(x - cp_3x).

    # Obtaining the control points that define the external arc of the hollow cylinder
    cp_ext1 = [radius_out, 0.0, 0.0]
    cp_ext3 = [radius_out * _np.cos(angle), radius_out * _np.sin(angle), 0.0]
    cp_ext2 = [
        radius_out,
        -(cp_ext3[0] / cp_ext3[1]) * (radius_out - cp_ext3[0]) + cp_ext3[1],
        0.0,
    ]

    # Obtaining the control points that define the internal arc of the hollow cylinder
    cp_int1 = [radius_in, 0.0, 0.0]
    cp_int3 = [radius_in * _np.cos(angle), radius_in * _np.sin(angle), 0.0]
    cp_int2 = [
        radius_in,
        -(cp_int3[0] / cp_int3[1]) * (radius_in - cp_int3[0]) + cp_int3[1],
        0.0,
    ]

    # Obtaining the control points positioned in the middle of the internal and external arches
    cp_middle1 = [(cp_ext1[0] + cp_int1[0]) / 2, (cp_ext1[1] + cp_int1[1]) / 2, 0.0]
    cp_middle2 = [(cp_ext2[0] + cp_int2[0]) / 2, (cp_ext2[1] + cp_int2[1]) / 2, 0.0]
    cp_middle3 = [(cp_ext3[0] + cp_int3[0]) / 2, (cp_ext3[1] + cp_int3[1]) / 2, 0.0]

    ctrlpts = [
        cp_ext1,
        cp_ext2,
        cp_ext3,
        cp_middle1,
        cp_middle2,
        cp_middle3,
        cp_int1,
        cp_int2,
        cp_int3,
    ]

    weights = [
        1.0,
        _np.cos(angle / 2),
        1.0,
        1.0,
        _np.cos(angle / 2),
        1.0,
        1.0,
        _np.cos(angle / 2),
        1.0,
    ]

    t_ctrlptsw = _compat.combine_ctrlpts_weights(ctrlpts, weights)
    n_ctrlptsw = _compat.flip_ctrlpts_u(t_ctrlptsw, p_size_u, p_size_v)

    surf.ctrlpts_size_u = p_size_u
    surf.ctrlpts_size_v = p_size_v
    surf.ctrlptsw = n_ctrlptsw

    surf.knotvector_u = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    surf.knotvector_v = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    do_uniform_knot_refinement_surface(surf, n_ele_u, n_ele_v)

    return surf


def create_nurbs_cylindrical_shell_sector(
    radius: float, angle: float, length: float, *, n_ele_u: int = 1, n_ele_v: int = 1
) -> _NURBS.Surface:
    """Creates a NURBS surface representing a 3D sector of a cylindrical shell.

    The center of the cylindrical shell sector is located at [0, 0, 0].

    Args:
        radius: Radius of the cylindrical shell.
        angle: Angle of the cylindrical shell sector in radians.
                       The angle is only valid for 0 < angle <= pi according
                       to "Isogeometric Analysis: Toward Integration of CAD and FEA" by
                       J.Austin Cottrell, 2009
        length: Length of the cylindrical shell.
        n_ele_u: Number of elements in the parametric u-direction. Defaults to 1.
        n_ele_v: Number of elements in the parametric v-direction. Defaults to 1.

    Returns:
        geomdl.NURBS.Surface: A geomdl object that contains the surface information.
    """
    # Check the validity of the input values:
    if (angle >= _np.pi) or (angle < 0):
        raise ValueError(
            "The following algorithm for creating a cylindrical shell sector is only valid for 0 < angle <= pi."
        )

    # Create a NURBS surface instance
    surf = _NURBS.Surface()

    # Set degrees
    surf.degree_u = 2
    surf.degree_v = 2

    # Control points and set them to the surface
    p_size_u = 3
    p_size_v = 3

    # Obtaining the control points
    cp_1 = [-radius * _np.sin(angle / 2), -length / 2, -radius * _np.cos(angle / 2)]
    cp_3 = [radius * _np.sin(angle / 2), -length / 2, -radius * _np.cos(angle / 2)]

    # Calculating position of middle points. This is done by
    # obtaining the tangents on the points cp_1 and cp_3 and
    # calculating the intersection point between the tangents.
    m_1 = -_np.tan(-angle / 2)
    m_3 = -_np.tan(angle / 2)
    b_1 = cp_1[2] + m_1 * cp_1[0]
    b_3 = cp_3[2] + m_3 * cp_3[0]

    inter_point_x = (b_3 - b_1) / (m_3 - m_1)
    inter_point_z = m_1 * inter_point_x + b_1

    # The intersection point is assigned to the middle points cp_2, cp_5 and cp_8
    cp_2 = [inter_point_x, -length / 2, inter_point_z]

    cp_4 = [-radius * _np.sin(angle / 2), 0.0, -radius * _np.cos(angle / 2)]
    cp_5 = [inter_point_x, 0.0, inter_point_z]
    cp_6 = [radius * _np.sin(angle / 2), 0.0, -radius * _np.cos(angle / 2)]

    cp_7 = [
        -radius * _np.sin(angle / 2),
        length / 2,
        -radius * _np.cos(angle / 2),
    ]
    cp_8 = [inter_point_x, length / 2, inter_point_z]
    cp_9 = [radius * _np.sin(angle / 2), length / 2, -radius * _np.cos(angle / 2)]

    ctrlpts = [cp_1, cp_4, cp_7, cp_2, cp_5, cp_8, cp_3, cp_6, cp_9]

    weights = [
        1.0,
        1.0,
        1.0,
        _np.cos(angle / 2),
        _np.cos(angle / 2),
        _np.cos(angle / 2),
        1.0,
        1.0,
        1.0,
    ]

    t_ctrlptsw = _compat.combine_ctrlpts_weights(ctrlpts, weights)

    surf.ctrlpts_size_u = p_size_u
    surf.ctrlpts_size_v = p_size_v
    surf.ctrlptsw = t_ctrlptsw

    surf.knotvector_u = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    surf.knotvector_v = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    do_uniform_knot_refinement_surface(surf, n_ele_u, n_ele_v)

    return surf


def create_nurbs_flat_plate_2d(width, length, *, n_ele_u=1, n_ele_v=1):
    """Creates a patch of a 2 dimensional flat plate.

    Args
    ----
    width: double
        dimension of the plate in the x-direction
    length: double
        dimension of the plate in the y-direction
    n_ele_u: int
        number of elements in the parametric u-direction
    n_ele_v: int
        number of elements in the parametric v-direction


    Return
    ----
    surf: geomdl object
        geomdl object that contains the surface information
    """
    # Create a NURBS surface instance
    surf = _NURBS.Surface()

    # Set degrees
    surf.degree_u = 2
    surf.degree_v = 2

    # Control points and set them to the surface
    p_size_u = 3
    p_size_v = 3

    ctrlpts = [
        [-width / 2, -length / 2, 0.0],
        [0.0, -length / 2, 0.0],
        [width / 2, -length / 2, 0.0],
        [-width / 2, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [width / 2, 0.0, 0.0],
        [-width / 2, length / 2, 0.0],
        [0.0, length / 2, 0.0],
        [width / 2, length / 2, 0.0],
    ]

    weights = [1.0] * 9

    t_ctrlptsw = _compat.combine_ctrlpts_weights(ctrlpts, weights)
    n_ctrlptsw = _compat.flip_ctrlpts_u(t_ctrlptsw, p_size_u, p_size_v)

    surf.ctrlpts_size_u = p_size_u
    surf.ctrlpts_size_v = p_size_v
    surf.ctrlptsw = n_ctrlptsw

    surf.knotvector_u = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    surf.knotvector_v = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    do_uniform_knot_refinement_surface(surf, n_ele_u, n_ele_v)

    return surf


def create_nurbs_sphere_surface(radius, n_ele_u=1, n_ele_v=1):
    """Generates a patch of a sphere as a NURBS surface.

    This function constructs a segment of a spherical surface using Non-Uniform Rational
    B-Splines (NURBS) based on the specified radius and the number of elements
    in the parametric u and v directions.

    Args
    ---
    radius: double
        radius of the sphere
    n_ele_u: int
        number of elements in the parametric u-direction
    n_ele_v: int
        number of elements in the parametric v-direction

    Return
    ----
    surf: geomdl object
        geomdl object that contains the surface information
    """
    # Create a NURBS surface instance
    surf = _NURBS.Surface()

    # Set degrees
    surf.degree_u = 2
    surf.degree_v = 2

    # Control points and set them to the surface
    p_size_u = 3
    p_size_v = 3

    dummy = 6.0 * (
        5.0 / 12.0
        + 0.5 * _np.sqrt(2.0 / 3.0)
        - 0.25 / _np.sqrt(3.0)
        - 0.5 * _np.sqrt(2.0 / 3.0) * _np.sqrt(3.0) / 2.0
    )

    ctrlpts = [
        [
            2.0 * radius / _np.sqrt(6.0) * -_np.sin(1.0 / 4.0 * _np.pi),
            -radius / _np.sqrt(3.0),
            2.0 * radius / _np.sqrt(6.0) * _np.cos(1.0 / 4.0 * _np.pi),
        ],
        [
            radius * _np.sqrt(3.0) / (_np.sqrt(2.0) * 2.0) * _np.cos(1.0 / 4.0 * _np.pi)
            + radius
            * _np.sqrt(3.0)
            / (_np.sqrt(2.0) * 2.0)
            * -_np.sin(1.0 / 4.0 * _np.pi),
            -_np.sqrt(3.0) / 2 * radius,
            radius * _np.sqrt(3.0) / (_np.sqrt(2.0) * 2.0) * _np.cos(1.0 / 4.0 * _np.pi)
            + radius
            * _np.sqrt(3.0)
            / (_np.sqrt(2.0) * 2.0)
            * _np.sin(1.0 / 4.0 * _np.pi),
        ],
        [
            2.0 * radius / _np.sqrt(6.0) * _np.cos(1.0 / 4.0 * _np.pi),
            -radius / _np.sqrt(3.0),
            2.0 * radius / _np.sqrt(6.0) * _np.sin(1.0 / 4.0 * _np.pi),
        ],
        [
            radius * _np.sqrt(6.0) / 2.0 * -_np.sin(1.0 / 4.0 * _np.pi),
            0.0,
            radius * _np.sqrt(6.0) / 2.0 * _np.cos(1.0 / 4.0 * _np.pi),
        ],
        [
            radius * dummy * _np.sqrt(2.0) / 2.0 * _np.cos(1.0 / 4.0 * _np.pi)
            + radius * dummy * _np.sqrt(2.0) / 2.0 * -_np.sin(1.0 / 4.0 * _np.pi),
            0.0,
            radius * dummy * _np.sqrt(2.0) / 2.0 * _np.cos(1.0 / 4.0 * _np.pi)
            + radius * dummy * _np.sqrt(2.0) / 2.0 * _np.sin(1.0 / 4.0 * _np.pi),
        ],
        [
            radius * _np.sqrt(6.0) / 2.0 * _np.cos(1.0 / 4.0 * _np.pi),
            0.0,
            radius * _np.sqrt(6.0) / 2.0 * _np.sin(1.0 / 4.0 * _np.pi),
        ],
        [
            2.0 * radius / _np.sqrt(6.0) * -_np.sin(1.0 / 4.0 * _np.pi),
            2.0 * radius / _np.sqrt(6.0) * _np.cos(1.0 / 4.0 * _np.pi),
            radius / _np.sqrt(3.0),
        ],
        [
            radius * _np.sqrt(3.0) / (_np.sqrt(2.0) * 2.0) * _np.cos(1.0 / 4.0 * _np.pi)
            + radius
            * _np.sqrt(3.0)
            / (_np.sqrt(2.0) * 2.0)
            * -_np.sin(1.0 / 4.0 * _np.pi),
            _np.sqrt(3.0) / 2 * radius,
            radius * _np.sqrt(3.0) / (_np.sqrt(2.0) * 2.0) * _np.cos(1.0 / 4.0 * _np.pi)
            + radius
            * _np.sqrt(3.0)
            / (_np.sqrt(2.0) * 2.0)
            * _np.sin(1.0 / 4.0 * _np.pi),
        ],
        [
            2.0 * radius / _np.sqrt(6.0) * _np.cos(1.0 / 4.0 * _np.pi),
            radius / _np.sqrt(3.0),
            2.0 * radius / _np.sqrt(6.0) * _np.sin(1.0 / 4.0 * _np.pi),
        ],
    ]

    weights = [
        1.0,
        2.0 / _np.sqrt(6.0),
        1.0,
        2.0 / _np.sqrt(6.0),
        2.0 / 3.0,
        2.0 / _np.sqrt(6.0),
        1.0,
        2.0 / _np.sqrt(6.0),
        1.0,
    ]

    t_ctrlptsw = _compat.combine_ctrlpts_weights(ctrlpts, weights)
    n_ctrlptsw = _compat.flip_ctrlpts_u(t_ctrlptsw, p_size_u, p_size_v)

    surf.ctrlpts_size_u = p_size_u
    surf.ctrlpts_size_v = p_size_v
    surf.ctrlptsw = n_ctrlptsw

    surf.knotvector_u = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    surf.knotvector_v = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    do_uniform_knot_refinement_surface(surf, n_ele_u, n_ele_v)

    return surf


def create_nurbs_hemisphere_surface(radius, n_ele_uv=1):
    """Generates a hemisphere as a NURBS surface.

    This function constructs five segments that represent the surface of a
    hemisphere using Non-Uniform Rational B-Splines (NURBS) based on the specified
    radius and the number of elements in the parametric u and v directions.
    To secure the connectivity between surfaces, all surfaces must have the same
    parametric representation in any parametric direction. Therefore, the number
    of elements in u- and v-directions must be the same.

    This function generates a list of five NURBS geomdl objects.

    Args
    ---
    radius: double
        Radius of the hemisphere
    n_ele_uv: int
        Number of elements in the parametric u- and v- directions

    Return
    ----
    list: list(geomdl object)
        A list of geomdl objects that contains the surface information
    """
    # Create the first section of the hemisphere
    hemisphere_1 = create_nurbs_sphere_surface(radius, n_ele_u=1, n_ele_v=1)

    # Create a temporary section by rotating and translating the
    # first section of the hemisphere
    temp_hemisphere = _operations.rotate(hemisphere_1, 90, axis=1)
    temp_hemisphere = _operations.translate(
        temp_hemisphere,
        (0, 0, -2.0 * radius / _np.sqrt(6.0) * _np.sin(1.0 / 4.0 * _np.pi) * 2),
    )

    # To create the hemisphere it is necessary to split the temporary
    # sphere section in two pieces. This split is done in u = 0.5. After
    # the split, the second surface is taken as it will be
    # adjacent to the first section of the sphere
    cut_section_sphere = _operations.split_surface_u(temp_hemisphere, param=0.5)
    hemisphere_2 = cut_section_sphere[1]

    translation_component = radius * _np.sin(1.0 / 4.0 * _np.pi) * 2
    # Create the third section. Rotate and translate it accordingly
    hemisphere_3 = _operations.rotate(hemisphere_2, 90, axis=2)
    hemisphere_3 = _operations.translate(hemisphere_3, (translation_component, 0, 0))

    # Create the forth section. Rotate and translate it accordingly
    hemisphere_4 = _operations.rotate(hemisphere_3, 90, axis=2)
    hemisphere_4 = _operations.translate(hemisphere_4, (0, translation_component, 0))

    # Create the fifth section. Rotate and translate it accordingly
    hemisphere_5 = _operations.rotate(hemisphere_4, 90, axis=2)
    hemisphere_5 = _operations.translate(hemisphere_5, (-translation_component, 0, 0))

    patches = [hemisphere_1, hemisphere_2, hemisphere_3, hemisphere_4, hemisphere_5]
    for patch in patches:
        do_uniform_knot_refinement_surface(patch, n_ele_uv, n_ele_uv)
    return patches


def create_nurbs_torus_surface(radius_torus, radius_circle, *, n_ele_u=1, n_ele_v=1):
    """Creates a NURBS patch for the construction of a ring torus (outer
    surface). A ring torus is a surface of revolution generated by revolving a
    circle in a 3-dimensional space around an axis of revolution (axis z). The
    center of the torus is located at the coordinates [0, 0, 0].

    This function constructs the surface of a ring torus in the following manner:
     - The quarter of the torus is made by 4 patches
     - These patches are rotated three times around the z-axis by 90°, e.g. theta = 90°, 180°, 270°,
       leading to a total of 16 NURBS patches.

    The number of elements in this function are given for a single patch, where:
     - n_ele_u is the number of elements in the "radius_torus" direction
     - n_ele_v is the number of elements in the "radius_circle" direction

    This function generates a list of 16 NURBS geomdl objects.

    Args
    ----
    radius_torus: double
        distance from the axis of revolution (axis z) to the center of the revolving circle
    radius_circle: double
        radius of the circle that revolves around the axis of revolution
    n_ele_u: int
        number of elements in the parametric u-direction
    n_ele_v: int
        number of elements in the parametric v-direction
    """

    # Create four NURBS surface instances. These are the base patches of the torus.
    surf_1 = _NURBS.Surface()
    surf_2 = _NURBS.Surface()
    surf_3 = _NURBS.Surface()
    surf_4 = _NURBS.Surface()
    base_surfs = [surf_1, surf_2, surf_3, surf_4]

    # Define control points and set them to the surfaces
    p_size_u = 3
    p_size_v = 3

    dummy_surf1 = radius_torus + radius_circle
    dummy_surf2 = radius_torus - radius_circle

    ctrlpts_surf1 = [
        [dummy_surf1, 0.0, 0.0],
        [dummy_surf1, 0.0, radius_circle],
        [radius_torus, 0.0, radius_circle],
        [dummy_surf1, dummy_surf1, 0.0],
        [dummy_surf1, dummy_surf1, radius_circle],
        [radius_torus, radius_torus, radius_circle],
        [0.0, dummy_surf1, 0.0],
        [0.0, dummy_surf1, radius_circle],
        [0.0, radius_torus, radius_circle],
    ]

    ctrlpts_surf2 = [
        [dummy_surf1, 0.0, 0.0],
        [dummy_surf1, 0.0, -radius_circle],
        [radius_torus, 0.0, -radius_circle],
        [dummy_surf1, dummy_surf1, 0.0],
        [dummy_surf1, dummy_surf1, -radius_circle],
        [radius_torus, radius_torus, -radius_circle],
        [0.0, dummy_surf1, 0.0],
        [0.0, dummy_surf1, -radius_circle],
        [0.0, radius_torus, -radius_circle],
    ]

    ctrlpts_surf3 = [
        [radius_torus, 0.0, -radius_circle],
        [dummy_surf2, 0.0, -radius_circle],
        [dummy_surf2, 0.0, 0.0],
        [radius_torus, radius_torus, -radius_circle],
        [dummy_surf2, dummy_surf2, -radius_circle],
        [dummy_surf2, dummy_surf2, 0.0],
        [0.0, radius_torus, -radius_circle],
        [0.0, dummy_surf2, -radius_circle],
        [0.0, dummy_surf2, 0.0],
    ]

    ctrlpts_surf4 = [
        [0.0, dummy_surf2, 0.0],
        [0.0, dummy_surf2, radius_circle],
        [0.0, radius_torus, radius_circle],
        [dummy_surf2, dummy_surf2, 0.0],
        [dummy_surf2, dummy_surf2, radius_circle],
        [radius_torus, radius_torus, radius_circle],
        [dummy_surf2, 0.0, 0.0],
        [dummy_surf2, 0.0, radius_circle],
        [radius_torus, 0.0, radius_circle],
    ]

    weights = [
        1.0,
        _np.sqrt(2) / 2,
        1.0,
        _np.sqrt(2) / 2,
        0.5,
        _np.sqrt(2) / 2,
        1.0,
        _np.sqrt(2) / 2,
        1.0,
    ]

    t_ctrlptsw1 = _compat.combine_ctrlpts_weights(ctrlpts_surf1, weights)
    t_ctrlptsw2 = _compat.combine_ctrlpts_weights(ctrlpts_surf2, weights)
    t_ctrlptsw3 = _compat.combine_ctrlpts_weights(ctrlpts_surf3, weights)
    t_ctrlptsw4 = _compat.combine_ctrlpts_weights(ctrlpts_surf4, weights)

    t_ctrlpts_surfs = [t_ctrlptsw1, t_ctrlptsw2, t_ctrlptsw3, t_ctrlptsw4]

    for surf, t_ctrlpts in zip(base_surfs, t_ctrlpts_surfs):
        # Set degrees
        surf.degree_u = 2
        surf.degree_v = 2

        surf.ctrlpts_size_u = p_size_u
        surf.ctrlpts_size_v = p_size_v

        # Set control points
        surf.ctrlptsw = t_ctrlpts

        # Set knot vectors
        surf.knotvector_u = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        surf.knotvector_v = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

        do_uniform_knot_refinement_surface(surf, n_ele_u, n_ele_v)

    # Define the rotations and translations to rotate the base patches and form a complete torus
    tmp_trans = [
        radius_torus + radius_circle,
        radius_torus + radius_circle,
        radius_torus,
        radius_torus - radius_circle,
    ]

    transform_surf1 = [
        [(-tmp_trans[0], tmp_trans[0], 0), 90, 2],
        [(-2 * tmp_trans[0], 0, 0), 180, 2],
        [(-tmp_trans[0], -tmp_trans[0], 0), 270, 2],
    ]

    transform_surf2 = [
        [(-tmp_trans[1], tmp_trans[1], 0), 90, 2],
        [(-2 * tmp_trans[1], 0, 0), 180, 2],
        [(-tmp_trans[1], -tmp_trans[1], 0), 270, 2],
    ]

    transform_surf3 = [
        [(-tmp_trans[2], tmp_trans[2], 0), 90, 2],
        [(-2 * tmp_trans[2], 0, 0), 180, 2],
        [(-tmp_trans[2], -tmp_trans[2], 0), 270, 2],
    ]

    transform_surf4 = [
        [(-tmp_trans[3], -tmp_trans[3], 0), 90, 2],
        [(0, -2 * tmp_trans[3], 0), 180, 2],
        [(tmp_trans[3], -tmp_trans[3], 0), 270, 2],
    ]

    # Rotate base patches and store them
    surfaces_torus = [surf_1, surf_2, surf_3, surf_4]
    for transform1, transform2, transform3, transform4 in zip(
        transform_surf1, transform_surf2, transform_surf3, transform_surf4
    ):
        new_surf1 = _operations.translate(surf_1, transform1[0])
        new_surf1 = _operations.rotate(new_surf1, transform1[1], axis=transform1[2])
        surfaces_torus.append(new_surf1)

        new_surf2 = _operations.translate(surf_2, transform2[0])
        new_surf2 = _operations.rotate(new_surf2, transform2[1], axis=transform2[2])
        surfaces_torus.append(new_surf2)

        new_surf3 = _operations.translate(surf_3, transform3[0])
        new_surf3 = _operations.rotate(new_surf3, transform3[1], axis=transform3[2])
        surfaces_torus.append(new_surf3)

        new_surf4 = _operations.translate(surf_4, transform4[0])
        new_surf4 = _operations.rotate(new_surf4, transform4[1], axis=transform4[2])
        surfaces_torus.append(new_surf4)

    return surfaces_torus


def create_nurbs_brick(width, length, height, *, n_ele_u=1, n_ele_v=1, n_ele_w=1):
    """Creates a patch of a 3 dimensional brick.

    Args
    ----
    width: double
        dimension of the plate in the x-direction
    length: double
        dimension of the plate in the y-direction
    height: double
        dimension of the plate in the z-direction
    n_ele_u: int
        number of elements in the parametric u-direction
    n_ele_v: int
        number of elements in the parametric v-direction
    n_ele_w: int
        number of elements in the parametric w-direction


    Return
    ----
    vol: geomdl object
        geomdl object that contains the volume information
    """
    # Create a NURBS volume instance
    vol = _NURBS.Volume()

    # Set degrees
    vol.degree_u = 2
    vol.degree_v = 2
    vol.degree_w = 2

    # Create control points and set them to the volume
    cp_size_u = 3
    cp_size_v = 3
    cp_size_w = 3

    ctrlpts = [
        [-width / 2, -length / 2, -height / 2],
        [-width / 2, 0.0, -height / 2],
        [-width / 2, length / 2, -height / 2],
        [0.0, -length / 2, -height / 2],
        [0.0, 0.0, -height / 2],
        [0.0, length / 2, -height / 2],
        [width / 2, -length / 2, -height / 2],
        [width / 2, 0.0, -height / 2],
        [width / 2, length / 2, -height / 2],
        [-width / 2, -length / 2, 0.0],
        [-width / 2, 0.0, 0.0],
        [-width / 2, length / 2, 0.0],
        [0.0, -length / 2, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, length / 2, 0.0],
        [width / 2, -length / 2, 0.0],
        [width / 2, 0.0, 0.0],
        [width / 2, length / 2, 0.0],
        [-width / 2, -length / 2, height / 2],
        [-width / 2, 0.0, height / 2],
        [-width / 2, length / 2, height / 2],
        [0.0, -length / 2, height / 2],
        [0.0, 0.0, height / 2],
        [0.0, length / 2, height / 2],
        [width / 2, -length / 2, height / 2],
        [width / 2, 0.0, height / 2],
        [width / 2, length / 2, height / 2],
    ]

    weights = [1.0] * 27

    vol.ctrlpts_size_u = cp_size_u
    vol.ctrlpts_size_v = cp_size_v
    vol.ctrlpts_size_w = cp_size_w

    t_ctrlptsw = _compat.combine_ctrlpts_weights(ctrlpts, weights)

    vol.set_ctrlpts(t_ctrlptsw, cp_size_u, cp_size_v, cp_size_w)

    vol.knotvector_u = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    vol.knotvector_v = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    vol.knotvector_w = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    do_uniform_knot_refinement_volume(vol, n_ele_u, n_ele_v, n_ele_w)

    return vol


def do_uniform_knot_refinement_surface(surf, n_ele_u, n_ele_v):
    """This function does an uniform knot refinement in the u- and v- direction.

    Args
    ----
    surf: geomdl object
        geomdl object that contains the surface information
    n_ele_u: int
        number of elements in the parametric u-direction
    n_ele_v: int
        number of elements in the parametric v-direction

    Return
    ----
    surf: geomdl object
    """
    size_of_knotvector_u = 1 / n_ele_u
    size_of_knotvector_v = 1 / n_ele_v

    for i in range(1, n_ele_u):
        _operations.insert_knot(surf, [size_of_knotvector_u * i, None], [1, 0])
    for j in range(1, n_ele_v):
        _operations.insert_knot(surf, [None, size_of_knotvector_v * j], [0, 1])


def do_uniform_knot_refinement_volume(vol, n_ele_u, n_ele_v, n_ele_w):
    """This function does an uniform knot refinement in the u-, v- and w- direction.

    Args
    ----
    vol: geomdl object
        geomdl object that contains the volume information
    n_ele_u: int
        number of elements in the parametric u-direction
    n_ele_v: int
        number of elements in the parametric v-direction
    n_ele_w: int
        number of elements in the parametric w-direction

    Return
    ----
    vol: geomdl object
    """
    size_of_knotvector_u = 1 / n_ele_u
    size_of_knotvector_v = 1 / n_ele_v
    size_of_knotvector_w = 1 / n_ele_w

    for i in range(1, n_ele_u):
        _operations.insert_knot(vol, [size_of_knotvector_u * i, None, None], [1, 0, 0])
    for j in range(1, n_ele_v):
        _operations.insert_knot(vol, [None, size_of_knotvector_v * j, None], [0, 1, 0])
    for k in range(1, n_ele_w):
        _operations.insert_knot(vol, [None, None, size_of_knotvector_w * k], [0, 0, 1])
