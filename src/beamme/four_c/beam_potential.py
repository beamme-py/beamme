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
"""This file includes functions to ease the creation of input files using beam
interaction potentials."""

import numpy as _np

from beamme.core.boundary_condition import BoundaryCondition as _BoundaryCondition
from beamme.core.function import Function as _Function
from beamme.core.geometry_set import GeometrySet as _GeometrySet


class BeamPotential:
    """Class which provides functions for the usage of beam to beam potential
    interactions within 4C based on a potential law in form of a power law."""

    def __init__(
        self,
        *,
        pot_law_prefactor: float | int | list | _np.ndarray,
        pot_law_exponent: float | int | list | _np.ndarray,
        pot_law_line_charge_density: float | int | list | _np.ndarray,
        pot_law_line_charge_density_funcs: _Function | list | _np.ndarray | None,
    ):
        """Initialize object to enable beam potential interactions.

        Args:
            pot_law_prefactors:
                Prefactors of a potential law in form of a power law. Same number
                of prefactors and exponents/line charge densities/functions must be
                provided!
            pot_law_exponent:
                Exponents of a potential law in form of a power law. Same number
                of exponents and prefactors/line charge densities/functions must be
                provided!
            pot_law_line_charge_density:
                Line charge densities of a potential law in form of a power law.
                Same number of line charge densities and prefactors/exponents/functions
                must be provided!
            pot_law_line_charge_density_funcs:
                Functions for line charge densities of a potential law in form of a
                power law. Same number of functions and prefactors/exponents/line
                charge densities must be provided!
        """

        # if only one potential law prefactor/exponent is present, convert it
        # into a list for simplified usage
        if isinstance(pot_law_prefactor, (float, int)):
            pot_law_prefactor = [pot_law_prefactor]
        if isinstance(pot_law_exponent, (float, int)):
            pot_law_exponent = [pot_law_exponent]
        if isinstance(pot_law_line_charge_density, (float, int)):
            pot_law_line_charge_density = [pot_law_line_charge_density]
        if (
            isinstance(pot_law_line_charge_density_funcs, _Function)
            or pot_law_line_charge_density_funcs is None
        ):
            pot_law_line_charge_density_funcs = [pot_law_line_charge_density_funcs]

        # check if same number of prefactors and exponents are provided
        if (
            not len(pot_law_prefactor)
            == len(pot_law_exponent)
            == len(pot_law_line_charge_density)
        ):
            raise ValueError(
                "Number of potential law prefactors do not match potential law exponents or potential line charge density!"
            )

        self.pot_law_prefactor = pot_law_prefactor
        self.pot_law_exponent = pot_law_exponent
        self.pot_law_line_charge_density = pot_law_line_charge_density
        self.pot_law_line_charge_density_funcs = pot_law_line_charge_density_funcs

    def create_header(
        self,
        *,
        potential_type: str,
        evaluation_strategy: str,
        cutoff_radius: float,
        regularization_type: str | None = None,
        regularization_separation: float,
        integration_segments: int,
        gauss_points: int,
        potential_reduction_length: float | int | None = None,
        automatic_differentiation: bool,
        choice_master_slave: str | None,
        runtime_output_interval_steps: int | None = None,
        runtime_output_every_iteration: bool,
        runtime_output_force: bool,
        runtime_output_moment: bool,
        runtime_output_uids: bool,
        runtime_output_per_ele_pair: bool,
    ) -> dict:
        """Set the basic header options for beam potential interactions.

        Args:
            potential_type:
                Type of applied potential.
            evaluation_strategy:
                Strategy to evaluate interaction potential.
            cutoff_radius:
                Neglect all contributions at separation larger than this cutoff
                radius.
            regularization_type:
                Type of regularization to use for force law at separations below
                specified separation.
            regularization_separation:
                Use specified regularization type for separations smaller than
                this value.
            integration_segments:
                Number of integration segments to be used per beam element.
            gauss_points:
                Number of Gauss points to be used per integration segment.
            potential_reduction_length:
                Potential is smoothly decreased within this length when using the
                single length specific (SBIP) approach to enable an axial pull off
                force.
            automatic_differentiation:
                Use automatic differentiation via FAD.
            choice_master_slave:
                Rule how to assign the role of master and slave to beam elements (if
                applicable).

            runtime_output:
                If the output for beam potential should be written.
            runtime_output_interval_steps:
                Interval at which output is written.
            runtime_output_every_iteration:
                If output at every Newton iteration should be written.
            runtime_output_force:
                If the forces should be written.
            runtime_output_moment:
                If the moments should be written.
            runtime_output_uids:
                If the unique ids should be written.
            runtime_output_per_ele_pair:
                If the forces/moments should be written per element pair.

        Returns:
            Header for beam potential interactions.
        """

        header = {
            "beam_potential": {
                "type": potential_type,
                "strategy": evaluation_strategy,
                "potential_law_prefactors": self.pot_law_prefactor,
                "potential_law_exponents": self.pot_law_exponent,
                "automatic_differentiation": automatic_differentiation,
                "cutoff_radius": cutoff_radius,
                "n_integration_segments": integration_segments,
                "n_gauss_points": gauss_points,
                "potential_reduction_length": potential_reduction_length,
            }
        }

        if regularization_type is not None:
            header["beam_potential"]["regularization"] = {
                "type": regularization_type,
                "separation": regularization_separation,
            }

        if choice_master_slave is not None:
            header["beam_potential"]["choice_master_slave"] = choice_master_slave

        if runtime_output_interval_steps is not None:
            header["beam_potential"]["runtime_output"] = {
                "interval_steps": runtime_output_interval_steps,
                "force": runtime_output_force,
                "moment": runtime_output_moment,
                "every_iteration": runtime_output_every_iteration,
                "write_force_moment_per_elementpair": runtime_output_per_ele_pair,
                "write_uids": runtime_output_uids,
            }

        return header

    def create_potential_charge_conditions(
        self, *, geometry_set: _GeometrySet
    ) -> list[_BoundaryCondition]:
        """Create potential charge conditions.

        Args:
            geometry_set:
                Add potential charge condition to this set.

        Returns:
            List of boundary conditions for potential charge.
        """

        bcs = []

        for i, (line_charge, func) in enumerate(
            zip(
                self.pot_law_line_charge_density, self.pot_law_line_charge_density_funcs
            )
        ):
            bc = _BoundaryCondition(
                geometry_set,
                {"POTLAW": i + 1, "VAL": line_charge, "FUNCT": func},
                bc_type="DESIGN LINE BEAM POTENTIAL CHARGE CONDITIONS",
            )

            bcs.append(bc)

        return bcs
