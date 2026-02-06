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
"""This file provides generators for commonly used test objects."""

from typing import Callable, Dict

import autograd.numpy as npAD
import numpy as np
import pytest
import splinepy

from beamme.core.material import MaterialBeamBase
from beamme.cosserat_curve.cosserat_curve import CosseratCurve
from beamme.four_c.material import (
    MaterialReissner,
    MaterialSolid,
    MaterialStVenantKirchhoff,
)


@pytest.fixture(scope="function")
def get_bc_data() -> Callable:
    """Return a function to create a dummy definition for a boundary condition
    in 4C.

    Returns:
        A function to create a dummy boundary condition definition.
    """

    def _get_bc_data(*, identifier=None, num_dof: int = 3) -> Dict:
        """Return a dummy definition for a boundary condition in 4C that can be
        used for testing purposes.

        Args:
            identifier: Any value, will be written to the value for the first DOF. This can be used to create multiple boundary conditions and distinguish them in the input file.
            num_dof: Number of degrees of freedom constrained by this boundary condition.
        """

        val = [0] * num_dof
        if identifier is not None:
            val[0] = identifier

        return {
            "NUMDOF": num_dof,
            "ONOFF": [1] * num_dof,
            "VAL": val,
            "FUNCT": [0] * num_dof,
        }

    return _get_bc_data


@pytest.fixture(scope="function")
def get_default_test_beam_material() -> Callable:
    """Return a function to create a default beam material for testing
    purposes.

    Returns:
        A function that creates a default beam material.
    """

    def _get_default_test_beam_material(material_type: str = "base", **kwargs):
        """Return a default material for testing purposes.

        Args:
            material_type: The type of beam material to return.

        Returns:
            A material object corresponding to the specified beam type.
        """

        if material_type == "base":
            return MaterialBeamBase(radius=1.0, **kwargs)

        elif material_type == "reissner":
            return MaterialReissner(
                radius=1.0, youngs_modulus=1.0, nu=0.3, density=1.0, **kwargs
            )

        else:
            raise ValueError(f"Unknown beam type: {material_type}")

    return _get_default_test_beam_material


@pytest.fixture(scope="function")
def get_default_test_solid_material() -> Callable:
    """Return a function to create a default solid material for testing
    purposes.

    Returns:
        A function that creates a default solid material.
    """

    def _get_default_test_solid_material(
        material_type: str = "st_venant_kirchhoff",
    ):
        """Return a default solid material for testing purposes.

        Args:
            material_type: The type of solid material to return.

        Returns:
            A material object corresponding to the specified solid material type.
        """

        if material_type == "st_venant_kirchhoff":
            return MaterialStVenantKirchhoff(youngs_modulus=1.0, nu=0.3, density=1.0)

        elif material_type == "2d_shell":
            return MaterialSolid(
                material_string="MAT_Kirchhoff_Love_shell",
                data={"YOUNG_MODULUS": 10.0, "POISSON_RATIO": 0.3, "THICKNESS": 0.05},
            )

        elif material_type == "solid_nested":
            return MaterialSolid(
                material_string="MAT_ElastHyper",
                data={
                    "NUMMAT": 2,
                    "MATIDS": [
                        MaterialSolid(
                            material_string="ELAST_CoupSVK",
                            data={"YOUNG": 1.0, "NUE": 0.0},
                        ),
                        MaterialSolid(
                            material_string="MAT_ElastHyper",
                            data={
                                "NUMMAT": 1,
                                "MATIDS": [
                                    MaterialSolid(
                                        material_string="ELAST_CoupSVK",
                                        data={"YOUNG": 2.0, "NUE": 0.0},
                                    )
                                ],
                                "DENS": 1.0,
                            },
                        ),
                    ],
                    "DENS": 1.0,
                },
            )

        else:
            raise ValueError(f"Unknown solid material type: {material_type}")

    return _get_default_test_solid_material


@pytest.fixture(scope="function")
def get_default_test_solid_element_description() -> Callable:
    """Return a function to create a default solid element description for
    testing purposes.

    Returns:
        A function that creates a default solid element description.
    """

    def _get_default_test_solid_element_description(
        element_type: str = "2d_solid",
    ):
        """Return a default solid element description for testing purposes.

        Args:
            element_type: The type of solid element to return.

        Returns:
            A dictionary containing the solid element description parameters.
        """

        if element_type == "2d_solid":
            return {
                "KINEM": "nonlinear",
                "EAS": "none",
                "THICK": 1.0,
                "STRESS_STRAIN": "plane_strain",
                "GP": [3, 3],
            }

        elif element_type == "2d_shell":
            return {"type": "SHELL_KIRCHHOFF_LOVE_NURBS", "GP": [3, 3]}

        elif element_type == "3d_solid":
            return {"KINEM": "nonlinear"}

        else:
            raise ValueError(f"Unknown solid element type: {element_type}")

    return _get_default_test_solid_element_description


@pytest.fixture(scope="function")
def get_parametric_function() -> Callable:
    """Fixture to create a function generator for parametric curves.

    Returns:
        A function generator that creates parametric curve functions for testing purposes.
    """

    def get_helix_function(
        radius: float,
        incline: float,
        *,
        transformation_factor: float | None = None,
        number_of_turns: float | None = None,
    ) -> Callable[[float], npAD.ndarray]:
        """Create and return a parametric function that represents a helix
        shape. The parameter coordinate can optionally be stretched to make the
        curve arc-length along the parameter coordinated non-constant and
        create a more complex curve for testing purposes.

        Args:
            radius: Radius of the helix
            incline: Incline of the helix
            transformation_factor: Factor to control the coordinate stretching (no direct physical interpretation)
            number_of_turns: Number of turns the helix will have to get approximate boundaries for the transformation.
                This is only used for the transformation, not the actual geometry, as we return the
                function to create the geometry and not the geometry itself.

        Returns:
            A function that describes a helix in 3D space.
        """

        if transformation_factor is None and number_of_turns is None:

            def transformation(t):
                """Return identity transformation."""
                return t

        elif transformation_factor is not None and number_of_turns is not None:

            def transformation(t):
                """Transform the parameter coordinate to make the function more
                complex."""
                return (
                    npAD.exp(
                        transformation_factor * t / (2.0 * np.pi * number_of_turns)
                    )
                    * t
                    / npAD.exp(transformation_factor)
                )

        else:
            raise ValueError(
                "You have to set none or both optional parameters: "
                "transformation_factor and number_of_turns"
            )

        def helix(t):
            """Parametric function to describe a helix."""
            return npAD.array(
                [
                    radius * npAD.cos(transformation(t)),
                    radius * npAD.sin(transformation(t)),
                    transformation(t) * incline / (2 * np.pi),
                ]
            )

        return helix

    def distorted_helix(t: float):
        """Parametric function to describe a distorted helix.

        The resulting curve has a large variation in the Jacobian along
        the curve, thus making it a good test case for the curve
        integration robustness and performance.
        """
        return npAD.array([t, 10 * npAD.sin(t), npAD.cos(t)])

    def _get_parametric_function(function_type: str, *args, **kwargs) -> Callable:
        """Return a function representing a parametric curve for testing
        purposes.

        Args:
            function_type: The type of parametric function to create.
            args: Positional arguments for the function generator.
            kwargs: Keyword arguments for the function generator.

        Returns:
            A function that creates a parametric curve.
        """

        if function_type == "helix":
            return get_helix_function(*args, **kwargs)
        elif function_type == "distorted_helix":
            return distorted_helix
        else:
            raise ValueError(f"Unknown parametric function type: {function_type}")

    return _get_parametric_function


@pytest.fixture(scope="function")
def get_splinepy_object() -> Callable:
    """Fixture that creates splinepy objects for testing purposes."""

    def _get_splinepy_object(splinepy_type: str) -> splinepy.Spline:
        """Create a splinepy object for testing purposes.

        Args:
            splinepy_type: The type of splinepy object to create.

        Returns:
            A splinepy object.
        """

        if splinepy_type == "curve_bezier":
            control_points = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 2.0, 1.0],
                    [2.0, 2.0, 2.0],
                    [3.0, 0.0, 0.0],
                ]
            )
            return splinepy.Bezier(degrees=[3], control_points=control_points)

        elif splinepy_type == "curve_nurbs":
            return splinepy.NURBS(
                degrees=[2],
                knot_vectors=[[0, 0, 0, 1, 1, 1]],
                control_points=[[0, 0, 0], [1, 2, -1], [2, 0, 0]],
                weights=[[1.0], [1.0], [1.0]],
            )

        elif splinepy_type == "curve_nurbs_distorted":
            # A NURBS curve that has a large variation in the Jacobian along the curve.
            return splinepy.NURBS(
                degrees=[2],
                knot_vectors=[[0, 0, 0, 1, 1, 1]],
                control_points=[[0, 0, 0], [1, 2, -1], [-6, 3, 10]],
                weights=[[1.0], [0.1], [2.0]],
            )

        else:
            raise ValueError(f"Unknown splinepy object type: {splinepy_type}")

    return _get_splinepy_object


@pytest.fixture(scope="function")
def get_cosserat_curve() -> Callable:
    """Fixture that creates a Cosserat curve from a predefined set of point
    coordinates."""

    def _get_cosserat_curve(**kwargs) -> CosseratCurve:
        """Create a Cosserat curve from a predefined set of point
        coordinates."""
        point_coordinates = np.array(
            [
                [
                    -7.125056969077852465e00,
                    -2.100498338589324305e02,
                    -1.315032467171957364e02,
                ],
                [
                    -8.578164819865914836e00,
                    -2.103983440417262045e02,
                    -1.318355861462153484e02,
                ],
                [
                    -1.002660147770902199e01,
                    -2.107622027327942078e02,
                    -1.321036387467290467e02,
                ],
                [
                    -1.420180257868918972e01,
                    -2.113479519751842020e02,
                    -1.338546926361781573e02,
                ],
                [
                    -1.548947274987304468e01,
                    -2.116635004519980043e02,
                    -1.345820241652732534e02,
                ],
                [
                    -1.669974530466243579e01,
                    -2.123023513258445121e02,
                    -1.352675684905443063e02,
                ],
                [
                    -1.780121899999767265e01,
                    -2.129407667828043600e02,
                    -1.359887265187280150e02,
                ],
                [
                    -1.875413870553265383e01,
                    -2.134432395695041862e02,
                    -1.368500440588985612e02,
                ],
                [
                    -1.951866843854408629e01,
                    -2.140215405181983783e02,
                    -1.379530359796511050e02,
                ],
                [
                    -2.005497221630869475e01,
                    -2.147109481986180981e02,
                    -1.392914797992089859e02,
                ],
                [
                    -2.032321405610320397e01,
                    -2.152536910592761785e02,
                    -1.407299275809807853e02,
                ],
            ]
        )
        return CosseratCurve(point_coordinates, **kwargs)

    return _get_cosserat_curve
