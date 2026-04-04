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
"""Generic function used to create NURBS meshes."""

import itertools as _itertools

import numpy as _np

from beamme.core.conf import bme as _bme
from beamme.core.geometry_set import GeometryName as _GeometryName
from beamme.core.geometry_set import GeometrySetNodes as _GeometrySetNodes
from beamme.core.mesh import Mesh as _Mesh
from beamme.core.node import ControlPoint as _ControlPoint
from beamme.core.nurbs_patch import NURBSSurface as _NURBSSurface
from beamme.core.nurbs_patch import NURBSVolume as _NURBSVolume


def _check_nurbs_dimension_and_element_type(
    nurbs_dimension: int, element_type: type
) -> None:
    """Check if the element type is compatible with the NURBS dimension.

    Args:
        nurbs_dimension: The dimension of the NURBS patch (2 for surface, 3 for volume).
        element_type: The type of element to be created.

    Raises:
        ValueError: If the element type is not compatible with the NURBS dimension.
    """
    if nurbs_dimension == 2 and not issubclass(element_type, _NURBSSurface):
        raise ValueError(
            "Error, expected element type to be a NURBSSurface for a NURBS surface!"
        )
    elif nurbs_dimension == 3 and not issubclass(element_type, _NURBSVolume):
        raise ValueError(
            "Error, expected element type to be a NURBSVolume for a NURBS volume!"
        )


def add_splinepy_nurbs_to_mesh(
    mesh: _Mesh, element_type: type, splinepy_obj, *, material=None
) -> _GeometryName:
    """Add a splinepy NURBS to the mesh.

    Args:
        mesh: Mesh that the created NURBS geometry will be added to.
        element_type: The type of element to be created.
        splinepy_obj (splinepy object): NURBS geometry created using splinepy.
        material (Material): Material for this geometry.

    Returns:
        GeometryName:
            Set with the control points that form the topology of the mesh.

            For a surface, the following information is stored:
                Vertices: 'vertex_u_min_v_min', 'vertex_u_max_v_min', 'vertex_u_min_v_max', 'vertex_u_max_v_max'
                Edges: 'line_v_min', 'line_u_max', 'line_v_max', 'line_u_min'
                Surface: 'surf'

            For a volume, the following information is stored:
                Vertices: 'vertex_u_min_v_min_w_min', 'vertex_u_max_v_min_w_min', 'vertex_u_min_v_max_w_min', 'vertex_u_max_v_max_w_min',
                        'vertex_u_min_v_min_w_max', 'vertex_u_max_v_min_w_max', 'vertex_u_min_v_max_w_max', 'vertex_u_max_v_max_w_max'
                Edges: 'line_v_min_w_min', 'line_u_max_w_min', 'line_v_max_w_min', 'line_u_min_w_min',
                    'line_u_min_v_min', 'line_u_max_v_min', 'line_u_min_v_max', 'line_u_max_v_max'
                    'line_v_min_w_max', 'line_u_max_w_max', 'line_v_max_w_max', 'line_u_min_w_max'
                Surfaces: 'surf_w_min', 'surf_w_max', 'surf_v_min', 'surf_v_max', 'surf_v_max', 'surf_u_min'
                Volume: 'vol'
    """

    # Make sure that the control points are 3D
    nurbs_cp_dim = splinepy_obj.control_points.shape[1]
    if not nurbs_cp_dim == 3:
        raise ValueError(f"Invalid control point dimension: {nurbs_cp_dim}")

    # Make sure the material is in the mesh
    mesh.add_material(material)

    # Fill control points
    control_points = [
        _ControlPoint(coord, weight[0])
        for coord, weight in zip(
            _np.asarray(splinepy_obj.control_points), _np.asarray(splinepy_obj.weights)
        )
    ]

    # Create elements
    _check_nurbs_dimension_and_element_type(
        len(splinepy_obj.knot_vectors), element_type
    )

    element = element_type(
        [_np.asarray(knot_vector) for knot_vector in splinepy_obj.knot_vectors],
        _np.asarray(splinepy_obj.degrees),
        nodes=control_points,
        material=material,
    )

    # Add element and control points to the mesh
    mesh.elements.append(element)
    mesh.nodes.extend(control_points)

    # Create geometry sets that will be returned
    return_set = create_geometry_sets(element)

    return return_set


def add_geomdl_nurbs_to_mesh(
    mesh: _Mesh, element_type: type, geomdl_obj, *, material=None
) -> _GeometryName:
    """Add a geomdl NURBS to the mesh.

    Args:
        mesh: Mesh that the created NURBS geometry will be added to.
        element_type: The type of element to be created.
        geomdl_obj (geomdl object): NURBS geometry created using geomdl.
        material (Material): Material for this geometry.

    Returns:
        GeometryName:
            Set with the control points that form the topology of the mesh.

            For a surface, the following information is stored:
                Vertices: 'vertex_u_min_v_min', 'vertex_u_max_v_min', 'vertex_u_min_v_max', 'vertex_u_max_v_max'
                Edges: 'line_v_min', 'line_u_max', 'line_v_max', 'line_u_min'
                Surface: 'surf'

            For a volume, the following information is stored:
                Vertices: 'vertex_u_min_v_min_w_min', 'vertex_u_max_v_min_w_min', 'vertex_u_min_v_max_w_min', 'vertex_u_max_v_max_w_min',
                        'vertex_u_min_v_min_w_max', 'vertex_u_max_v_min_w_max', 'vertex_u_min_v_max_w_max', 'vertex_u_max_v_max_w_max'
                Edges: 'line_v_min_w_min', 'line_u_max_w_min', 'line_v_max_w_min', 'line_u_min_w_min',
                    'line_u_min_v_min', 'line_u_max_v_min', 'line_u_min_v_max', 'line_u_max_v_max'
                    'line_v_min_w_max', 'line_u_max_w_max', 'line_v_max_w_max', 'line_u_min_w_max'
                Surfaces: 'surf_w_min', 'surf_w_max', 'surf_v_min', 'surf_v_max', 'surf_v_max', 'surf_u_min'
                Volume: 'vol'
    """

    # Make sure the material is in the mesh
    mesh.add_material(material)

    # Fill control points
    control_points = []
    nurbs_dimension = len(geomdl_obj.knotvector)
    if nurbs_dimension == 2:
        control_points = create_control_points_surface(geomdl_obj)
    elif nurbs_dimension == 3:
        control_points = create_control_points_volume(geomdl_obj)
    else:
        raise NotImplementedError(
            "Error, not implemented for NURBS with dimension {}!".format(
                nurbs_dimension
            )
        )

    # Create elements
    _check_nurbs_dimension_and_element_type(len(geomdl_obj.knotvector), element_type)
    element = element_type(
        geomdl_obj.knotvector,
        geomdl_obj.degree,
        nodes=control_points,
        material=material,
    )

    # Add element and control points to the mesh
    mesh.elements.append(element)
    mesh.nodes.extend(control_points)

    # Create geometry sets that will be returned
    return_set = create_geometry_sets(element)

    return return_set


def create_control_points_surface(geomdl_obj):
    """Creates a list with the ControlPoint objects of a surface created with
    geomdl."""
    control_points = []
    for dir_v in range(geomdl_obj.ctrlpts_size_v):
        for dir_u in range(geomdl_obj.ctrlpts_size_u):
            weight = geomdl_obj.ctrlpts2d[dir_u][dir_v][3]

            # As the control points are scaled with their weight, divide them to get
            # their coordinates
            coord = [
                geomdl_obj.ctrlpts2d[dir_u][dir_v][0] / weight,
                geomdl_obj.ctrlpts2d[dir_u][dir_v][1] / weight,
                geomdl_obj.ctrlpts2d[dir_u][dir_v][2] / weight,
            ]

            control_points.append(_ControlPoint(coord, weight))

    return control_points


def create_control_points_volume(geomdl_obj):
    """Creates a list with the ControlPoint objects of a volume created with
    geomdl."""
    control_points = []
    for dir_w in range(geomdl_obj.ctrlpts_size_w):
        for dir_v in range(geomdl_obj.ctrlpts_size_v):
            for dir_u in range(geomdl_obj.ctrlpts_size_u):
                # Obtain the id of the control point
                cp_id = (
                    dir_v
                    + geomdl_obj.ctrlpts_size_v * dir_u
                    + geomdl_obj.ctrlpts_size_u * geomdl_obj.ctrlpts_size_v * dir_w
                )

                weight = geomdl_obj.ctrlptsw[cp_id][3]

                # As the control points are scaled with their weight, divide them to get
                # their coordinates
                coord = [
                    geomdl_obj.ctrlptsw[cp_id][0] / weight,
                    geomdl_obj.ctrlptsw[cp_id][1] / weight,
                    geomdl_obj.ctrlptsw[cp_id][2] / weight,
                ]

                control_points.append(_ControlPoint(coord, weight))

    return control_points


def create_geometry_sets(element: _NURBSSurface | _NURBSVolume) -> _GeometryName:
    """Create the geometry sets for NURBS patches of all dimensions.

    Args:
        element: The NURBS patch for which the geometry sets should be created.

    Returns:
        The geometry set container for the given NURBS patch.
    """

    # Create return set
    return_set = _GeometryName()

    # Get general data needed for the set creation
    num_cps_uvw = element.get_number_of_control_points_per_dir()
    nurbs_dimension = len(element.knot_vectors)
    n_cp = _np.prod(num_cps_uvw)
    axes = ["u", "v", "w"][:nurbs_dimension]
    name_map = {0: "min", -1: "max", 1: "min_next", -2: "max_next"}

    # This is a tensor array that contains the CP indices
    cp_indices_dim = _np.arange(n_cp, dtype=int).reshape(*num_cps_uvw[::-1]).transpose()

    if nurbs_dimension >= 0:
        # Add point sets
        directions = [0, -1]
        for corner in _itertools.product(*([directions] * nurbs_dimension)):
            name = "vertex_" + "_".join(
                f"{axis}_{name_map[coord]}" for axis, coord in zip(axes, corner)
            )
            index = cp_indices_dim[corner]
            return_set[name] = _GeometrySetNodes(
                _bme.geo.point, nodes=element.nodes[index]
            )

    if nurbs_dimension > 0:
        # Add edge sets

        if nurbs_dimension == 2:
            directions = [0, 1, -2, -1]
        elif nurbs_dimension == 3:
            directions = [0, -1]
        else:
            raise ValueError("NURBS dimension 1 not implemented")

        # Iterate over each axis (the axis that varies — the "edge" direction)
        for edge_axis in range(nurbs_dimension):
            # The other axes will be fixed
            fixed_axes = [i for i in range(nurbs_dimension) if i != edge_axis]
            for fixed_dir in _itertools.product(*([directions] * len(fixed_axes))):
                # Build slicing tuple for indexing cp_indices_dim
                slicer: list[slice | int] = [slice(None)] * nurbs_dimension
                name_parts = []
                for axis_idx, dir_val in zip(fixed_axes, fixed_dir):
                    slicer[axis_idx] = dir_val
                    name_parts.append(f"{axes[axis_idx]}_{name_map[dir_val]}")
                name = "line_" + "_".join(name_parts)

                # Get node indices along the edge
                edge_indices = cp_indices_dim[tuple(slicer)].flatten()
                return_set[name] = _GeometrySetNodes(
                    _bme.geo.line, nodes=[element.nodes[i] for i in edge_indices]
                )

    if nurbs_dimension == 2:
        # Add surface sets for surface NURBS
        return_set["surf"] = _GeometrySetNodes(_bme.geo.surface, element.nodes)

    if nurbs_dimension == 3:
        # Add surface sets for volume NURBS
        for fixed_axis in range(nurbs_dimension):
            for dir_val in directions:
                # Build slice and name
                slicer = [slice(None)] * nurbs_dimension
                slicer[fixed_axis] = dir_val
                surface_name = f"surf_{axes[fixed_axis]}_{name_map[dir_val]}"

                surface_indices = cp_indices_dim[tuple(slicer)].flatten()
                return_set[surface_name] = _GeometrySetNodes(
                    _bme.geo.surface,
                    nodes=[element.nodes[i] for i in surface_indices],
                )

        # Add volume sets
        return_set["vol"] = _GeometrySetNodes(_bme.geo.volume, element.nodes)

    return return_set
