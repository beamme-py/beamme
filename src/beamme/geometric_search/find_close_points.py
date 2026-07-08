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
"""Find unique points in a point cloud, i.e., points that are within a certain tolerance
of each other will be considered as unique."""

from enum import Enum as _Enum
from enum import auto as _auto

from beamme.geometric_search.scipy import (
    find_close_points_scipy as _find_close_points_scipy,
)
from beamme.geometric_search.utils import arborx_is_available as _arborx_is_available
from beamme.geometric_search.utils import cython_is_available as _cython_is_available

if _cython_is_available():
    from beamme.geometric_search.cython import (
        find_close_points_brute_force_cython as _find_close_points_brute_force_cython,
    )

if _arborx_is_available():
    from beamme.geometric_search.arborx import (
        find_close_points_arborx as _find_close_points_arborx,
    )


class FindClosePointAlgorithm(_Enum):
    """Enum for different find_close_point algorithms."""

    kd_tree_scipy = _auto()
    brute_force_cython = _auto()
    boundary_volume_hierarchy_arborx = _auto()


def point_partners_to_unique_indices(point_partners):
    """Convert the partner indices to lists that can be used for converting between the
    full and unique coordinates.

    Returns
    ----
    unique_indices: list(int)
        Indices that result in the unique point coordinate array.
    inverse_indices: list(int)
        Indices of the unique array that can be used to reconstruct of the original points coordinates.
    """
    unique_indices = []
    inverse_indices = [-1 for i in range(len(point_partners))]
    partner_id_to_unique_map = {}
    i_partner = 0
    for i_point, partner_index in enumerate(point_partners):
        if partner_index == -1:
            # This point does not have any partners, i.e., it is already a unique point.
            unique_indices.append(i_point)
            my_inverse_index = len(unique_indices) - 1
        elif partner_index == i_partner:
            # This point has partners and this is the first time that this partner index
            # appears in the input list.
            unique_indices.append(i_point)
            my_inverse_index = len(unique_indices) - 1
            partner_id_to_unique_map[partner_index] = my_inverse_index
            i_partner += 1
        elif partner_index < i_partner:
            # This point has partners and the partner index has been previously found.
            my_inverse_index = partner_id_to_unique_map[partner_index]
        else:
            raise ValueError(
                "This should not happen, as the partners should be provided in order"
            )
        inverse_indices[i_point] = my_inverse_index

    return unique_indices, inverse_indices


def point_partners_to_partner_indices(point_partners, n_partners):
    """Convert the partner indices for each point to a list of lists with the indices
    for all partners."""
    partner_indices = [[] for i in range(n_partners)]
    for i, partner_index in enumerate(point_partners):
        if partner_index != -1:
            partner_indices[partner_index].append(i)
    return partner_indices


def partner_indices_to_point_partners(partner_indices, n_points):
    """Convert the list of lists with the indices for all partners to the partner
    indices for each point."""
    point_partners = [-1 for _i in range(n_points)]
    for i_partner, partners in enumerate(partner_indices):
        for index in partners:
            point_partners[index] = i_partner
    return point_partners, len(partner_indices)


def find_close_points(point_coordinates, *, algorithm=None, tol=1e-8, **kwargs):
    """Find unique points in a point cloud, i.e., points that are within a certain
    tolerance of each other will be considered as unique.

    Args
    ----
    point_coordinates: _np.array(n_points x n_dim)
        Point coordinates that are checked for partners. The number of spatial dimensions
        does not have to be equal to 3.
    algorithm: FindClosePointAlgorithm
        Type of geometric search algorithm that should be used.
    n_bins: list(int)
        Number of bins in the first three dimensions.
    tol: float
        If the absolute distance between two points is smaller than tol, they
        are considered to be equal, i.e., tol is the hyper sphere radius that
        the point coordinates have to be within, to be identified as overlapping.

    Return
    ----
    has_partner: array(int)
        An array with integers, marking the partner index of each point. A partner
        index of -1 means the node does not have a partner.
    partner: int
        Largest partner index.
    """
    n_points = len(point_coordinates)

    if algorithm is None:
        # Decide which algorithm to use
        if n_points < 200 and _cython_is_available():
            # For around 200 points the brute force cython algorithm is the fastest one
            algorithm = FindClosePointAlgorithm.brute_force_cython
        elif _arborx_is_available():
            # For general problems with n_points > 200 the ArborX implementation is the fastest one
            algorithm = FindClosePointAlgorithm.boundary_volume_hierarchy_arborx
        else:
            # The scipy implementation is slower than ArborX by a factor of about 2, but is scales
            # the same
            algorithm = FindClosePointAlgorithm.kd_tree_scipy

    # Get list of closest pairs
    if algorithm is FindClosePointAlgorithm.kd_tree_scipy:
        has_partner, n_partner = _find_close_points_scipy(
            point_coordinates, tol, **kwargs
        )
    elif algorithm is FindClosePointAlgorithm.brute_force_cython:
        has_partner, n_partner = _find_close_points_brute_force_cython(
            point_coordinates, tol, **kwargs
        )
    elif algorithm is FindClosePointAlgorithm.boundary_volume_hierarchy_arborx:
        has_partner, n_partner = _find_close_points_arborx(
            point_coordinates, tol, **kwargs
        )
    else:
        raise TypeError(f"Got unexpected algorithm {algorithm}")

    return has_partner, n_partner
