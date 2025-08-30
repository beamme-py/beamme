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

from typing import Any as _Any

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

    def dump_element_specific_section(self, input_file) -> None:
        """Set the knot vectors of the NURBS patch in the input file."""

        patch_data: dict[str, _Any] = {
            "KNOT_VECTORS": [],
        }

        for dir_manifold in range(self.get_nurbs_dimension()):
            knotvector = self.knot_vectors[dir_manifold]
            num_knots = len(knotvector)

            # Check the type of knot vector, in case that the multiplicity of the first and last
            # knot vectors is not p + 1, then it is a closed (periodic) knot vector, otherwise it
            # is an open (interpolated) knot vector.
            knotvector_type = "Interpolated"

            for i in range(self.polynomial_orders[dir_manifold] - 1):
                if (abs(knotvector[i] - knotvector[i + 1]) > _bme.eps_knot_vector) or (
                    abs(knotvector[num_knots - 2 - i] - knotvector[num_knots - 1 - i])
                    > _bme.eps_knot_vector
                ):
                    knotvector_type = "Periodic"
                    break

            patch_data["KNOT_VECTORS"].append(
                {
                    "DEGREE": self.polynomial_orders[dir_manifold],
                    "TYPE": knotvector_type,
                    "KNOTS": [
                        knot_vector_val
                        for knot_vector_val in self.knot_vectors[dir_manifold]
                    ],
                }
            )

        if "STRUCTURE KNOTVECTORS" in input_file:
            # Get all existing patches in the input file - they will be added to the
            # input file again at the end of this function. By doing it this way, the
            # FourCIPP type converter will be applied to the current patch.
            # This also means that we apply the type converter again already existing
            # patches. But, with the usual number of patches and data size, this
            # should not lead to a measurable performance impact.
            patches = input_file.pop("STRUCTURE KNOTVECTORS")["PATCHES"]
        else:
            patches = []

        patch_data["ID"] = self.i_nurbs_patch + 1
        patches.append(patch_data)
        input_file.add({"STRUCTURE KNOTVECTORS": {"PATCHES": patches}})

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

    def dump_to_list(self):
        """Return a list with all the element definitions contained in this
        patch."""

        # Check the material
        self._check_material()

        # Calculate how many control points are on the u direction
        ctrlpts_size_u = len(self.knot_vectors[0]) - self.polynomial_orders[0] - 1

        def get_ids_ctrlpts_surface(knot_span_u, knot_span_v):
            """For an interpolated patch, calculate control points involved in
            evaluation of the surface point at the knot span (knot_span_u,
            knot_span_v)"""

            id_u = knot_span_u - self.polynomial_orders[0]
            id_v = knot_span_v - self.polynomial_orders[1]

            element_ctrlpts_ids = []
            for j in range(self.polynomial_orders[1] + 1):
                for i in range(self.polynomial_orders[0] + 1):
                    # Calculating the global index of the control point, based on the book
                    # "Isogeometric Analysis: toward Integration of CAD and FEA" of J. Austin
                    # Cottrell, p. 314.
                    index_global = ctrlpts_size_u * (id_v + j) + id_u + i
                    element_ctrlpts_ids.append(index_global)

            return element_ctrlpts_ids

        patch_elements = []

        # Adding an increment j to self.global to obtain the ID of an element in the patch
        j = 0

        # Loop over the knot spans to obtain the elements inside the patch
        for knot_span_v in range(
            self.polynomial_orders[1],
            len(self.knot_vectors[1]) - self.polynomial_orders[1] - 1,
        ):
            for knot_span_u in range(
                self.polynomial_orders[0],
                len(self.knot_vectors[0]) - self.polynomial_orders[0] - 1,
            ):
                element_cps_ids = get_ids_ctrlpts_surface(knot_span_u, knot_span_v)

                connectivity = [self.nodes[i] for i in element_cps_ids]

                num_cp_in_element = (self.polynomial_orders[0] + 1) * (
                    self.polynomial_orders[1] + 1
                )

                patch_elements.append(
                    {
                        "id": self.i_global + j + 1,
                        "cell": {
                            "type": f"NURBS{num_cp_in_element}",
                            "connectivity": connectivity,
                        },
                        "data": {
                            "type": "WALLNURBS",
                            "MAT": self.material,
                            **(self.data if self.data else {}),
                        },
                    }
                )
                j += 1

        return patch_elements


class NURBSVolume(NURBSPatch):
    """A patch of a NURBS volume."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def dump_to_list(self):
        """Return a list with all the element definitions contained in this
        patch."""

        # Check the material
        self._check_material()

        # Calculate how many control points are on the u and v directions
        ctrlpts_size_u = len(self.knot_vectors[0]) - self.polynomial_orders[0] - 1
        ctrlpts_size_v = len(self.knot_vectors[1]) - self.polynomial_orders[1] - 1

        def get_ids_ctrlpts_volume(knot_span_u, knot_span_v, knot_span_w):
            """For an interpolated patch, calculate control points involved in
            evaluation of the surface point at the knot span (knot_span_u,
            knot_span_v, knot_span_w)"""

            id_u = knot_span_u - self.polynomial_orders[0]
            id_v = knot_span_v - self.polynomial_orders[1]
            id_w = knot_span_w - self.polynomial_orders[2]

            element_ctrlpts_ids = []

            for k in range(self.polynomial_orders[2] + 1):
                for j in range(self.polynomial_orders[1] + 1):
                    for i in range(self.polynomial_orders[0] + 1):
                        # Calculating the global index of the control point, based on the paper
                        # "Isogeometric analysis: an overview and computer implementation aspects"
                        # of Vinh-Phu Nguyen, Mathematics and Computers in Simulation, Jun-2015.
                        index_global = (
                            ctrlpts_size_u * ctrlpts_size_v * (id_w + k)
                            + ctrlpts_size_u * (id_v + j)
                            + id_u
                            + i
                        )
                        element_ctrlpts_ids.append(index_global)

            return element_ctrlpts_ids

        patch_elements = []

        # Adding an increment to self.global to obtain the ID of an element in the patch
        increment_ele = 0

        # Loop over the knot spans to obtain the elements inside the patch
        for knot_span_w in range(
            self.polynomial_orders[2],
            len(self.knot_vectors[2]) - self.polynomial_orders[2] - 1,
        ):
            for knot_span_v in range(
                self.polynomial_orders[1],
                len(self.knot_vectors[1]) - self.polynomial_orders[1] - 1,
            ):
                for knot_span_u in range(
                    self.polynomial_orders[0],
                    len(self.knot_vectors[0]) - self.polynomial_orders[0] - 1,
                ):
                    element_cps_ids = get_ids_ctrlpts_volume(
                        knot_span_u, knot_span_v, knot_span_w
                    )

                    connectivity = [self.nodes[i] for i in element_cps_ids]

                    num_cp_in_element = (
                        (self.polynomial_orders[0] + 1)
                        * (self.polynomial_orders[1] + 1)
                        * (self.polynomial_orders[2] + 1)
                    )

                    patch_elements.append(
                        {
                            "id": self.i_global + increment_ele + 1,
                            "cell": {
                                "type": f"NURBS{num_cp_in_element}",
                                "connectivity": connectivity,
                            },
                            "data": {
                                "type": "SOLID",
                                "MAT": self.material,
                                **(self.data if self.data else {}),
                            },
                        }
                    )
                    increment_ele += 1

        return patch_elements
