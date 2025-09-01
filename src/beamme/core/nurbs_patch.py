# The MIT License (MIT)
#
# Copyright (c) 2018-2025 BeamMe Authors
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
"""This module implements NURBS patches for the mesh."""

from typing import Iterator as _Iterator

import numpy as _np

from beamme.core.conf import bme as _bme
from beamme.core.element import Element as _Element
from beamme.core.material import (
    MaterialSolidBase as _MaterialSolidBase,
)


class NURBSPatch(_Element):
    """A base class for a NURBS patch."""

    # A list of valid material types for this element
    valid_materials = [_MaterialSolidBase]

    def __init__(
        self,
        knot_vectors,
        polynomial_orders,
        material=None,
        nodes=None,
        data=None,
    ):
        super().__init__(nodes=nodes, material=material, data=data)

        # Knot vectors
        self.knot_vectors = knot_vectors

        # Polynomial degrees
        self.polynomial_orders = polynomial_orders

        # Set numbers for elements
        self.i_nurbs_patch = None

    def get_nurbs_dimension(self) -> int:
        """Determine the number of dimensions of the NURBS structure.

        Returns:
            Number of dimensions of the NURBS object.
        """
        n_knots = len(self.knot_vectors)
        n_polynomial = len(self.polynomial_orders)
        if not n_knots == n_polynomial:
            raise ValueError(
                "The variables n_knots and polynomial_orders should have "
                f"the same length. Got {n_knots} and {n_polynomial}"
            )
        return n_knots

    def get_number_of_control_points_per_dir(self) -> list[int]:
        """Determine the number of control points in each parameter direction
        of the patch.

        Returns:
            List of control points per direction.
        """
        n_dim = len(self.knot_vectors)
        n_cp_per_dim = []
        for i_dim in range(n_dim):
            knot_vector_size = len(self.knot_vectors[i_dim])
            polynomial_order = self.polynomial_orders[i_dim]
            n_cp_per_dim.append(knot_vector_size - polynomial_order - 1)
        return n_cp_per_dim

    def get_number_elements(self) -> int:
        """Determine the number of elements in this patch by checking the
        amount of nonzero knot spans in the knot vector.

        Returns:
            Number of elements for this patch.
        """

        num_elements_dir = _np.zeros(len(self.knot_vectors), dtype=int)

        for i_dir in range(len(self.knot_vectors)):
            for i_knot in range(len(self.knot_vectors[i_dir]) - 1):
                if (
                    abs(
                        self.knot_vectors[i_dir][i_knot]
                        - self.knot_vectors[i_dir][i_knot + 1]
                    )
                    > _bme.eps_knot_vector
                ):
                    num_elements_dir[i_dir] += 1

        total_num_elements = _np.prod(num_elements_dir)

        return total_num_elements

    def _check_material(self) -> None:
        """Check if the linked material is valid for this type of NURBS solid
        element."""
        for material_type in type(self).valid_materials:
            if isinstance(self.material, material_type):
                return
        raise TypeError(
            f"NURBS solid of type {type(self)} can not have a material of"
            f" type {type(self.material)}!"
        )


class NURBSSurface(NURBSPatch):
    """A patch of a NURBS surface."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_knot_span_iterator(self) -> _Iterator[tuple[int, ...]]:
        """Return a tuple with the knot spans for this patch."""

        p, q = self.polynomial_orders
        kv_u, kv_v = self.knot_vectors[:2]
        return (
            (u, v)
            for v in range(q, len(kv_v) - q - 1)
            for u in range(p, len(kv_u) - p - 1)
        )

    def _get_ids_ctrlpts(self, knot_span_u: int, knot_span_v: int) -> list[int]:
        """Compute the global indices of the control points that influence the
        element defined by the given knot span."""

        p, q = self.polynomial_orders
        ctrlpts_size_u = len(self.knot_vectors[0]) - p - 1
        id_u = knot_span_u - p
        id_v = knot_span_v - q

        return [
            ctrlpts_size_u * (id_v + j) + id_u + i
            for j in range(q + 1)
            for i in range(p + 1)
        ]


class NURBSVolume(NURBSPatch):
    """A patch of a NURBS volume."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_knot_span_iterator(self) -> _Iterator[tuple[int, ...]]:
        """Return a tuple with the knot spans for this patch."""

        p, q, r = self.polynomial_orders
        kv_u, kv_v, kv_w = self.knot_vectors
        return (
            (u, v, w)
            for w in range(r, len(kv_w) - r - 1)
            for v in range(q, len(kv_v) - q - 1)
            for u in range(p, len(kv_u) - p - 1)
        )

    def _get_ids_ctrlpts(
        self, knot_span_u: int, knot_span_v: int, knot_span_w: int
    ) -> list[int]:
        """Compute the global indices of the control points that influence the
        element defined by the given knot span."""

        p, q, r = self.polynomial_orders
        id_u = knot_span_u - p
        id_v = knot_span_v - q
        id_w = knot_span_w - r
        size_u = len(self.knot_vectors[0]) - p - 1
        size_v = len(self.knot_vectors[1]) - q - 1

        return [
            size_u * size_v * (id_w + k) + size_u * (id_v + j) + id_u + i
            for k in range(r + 1)
            for j in range(q + 1)
            for i in range(p + 1)
        ]
