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

from beamme.core.boundary_condition import BoundaryCondition as _BoundaryCondition


class BeamPotential:
    """Class which provides functions for the usage of beam to beam potential
    interactions within 4C based on a potential law in form of a power law."""

    def __init__(
        self,
        input_file,
        mesh,
        *,
        pot_law_prefactor=None,
        pot_law_exponent=None,
        pot_law_line_charge_density=None,
        pot_law_line_charge_density_funcs=None,
    ):
        """Initialize object to enable beam potential interactions.

        Args
        ----
        input_file:
            Input file of current problem setup.
        mesh:
            Mesh object of current problem setup.
        pot_law_prefactors: float, int, _np.array, list
            Prefactors of a potential law in form of a power law. Same number
            of prefactors and exponents/line charge densities/functions must be
            provided!
        pot_law_exponent: float, int, _np.array, list
            Exponents of a potential law in form of a power law. Same number
            of exponents and prefactors/line charge densities/functions must be
            provided!
        pot_law_line_charge_density: float, int, _np.array, list
            Line charge densities of a potential law in form of a power law.
            Same number of line charge densities and prefactors/exponents/functions
            must be provided!
        pot_law_line_charge_density_funcs:
            Functions for line charge densities of a potential law in form of a
            power law. Same number of functions and prefactors/exponents/line
            charge densities must be provided!
        """

        self.input_file = input_file
        self.mesh = mesh

        # if only one potential law prefactor/exponent is present, convert it
        # into a list for simplified usage
        if isinstance(pot_law_prefactor, (float, int)):
            pot_law_prefactor = [pot_law_prefactor]
        if isinstance(pot_law_exponent, (float, int)):
            pot_law_exponent = [pot_law_exponent]
        if isinstance(pot_law_line_charge_density, (float, int)):
            pot_law_line_charge_density = [pot_law_line_charge_density]

        # check if same number of prefactors and exponents are provided
        if (
            not len(pot_law_prefactor)
            == len(pot_law_exponent)
            == len(pot_law_line_charge_density)
        ):
            raise ValueError(
                "Number of potential law prefactors do not match potential law exponents!"
            )

        self.pot_law_prefactor = pot_law_prefactor
        self.pot_law_exponent = pot_law_exponent
        self.pot_law_line_charge_density = pot_law_line_charge_density
        self.pot_law_line_charge_density_funcs = pot_law_line_charge_density_funcs

    def add_header(
        self,
        *,
        potential_type="volume",
        cutoff_radius=None,
        evaluation_strategy=None,
        regularization_type=None,
        regularization_separation=None,
        integration_segments=1,
        gauss_points=10,
        potential_reduction_length=-1,
        automatic_differentiation=False,
        choice_master_slave=None,
    ):
        """Set the basic header options for beam potential interactions.

        Args
        ----
        potential_type: string
            Type of applied potential (volume, surface).
        cutoff_radius: float
            Neglect all contributions at separation larger than this cutoff
            radius.
        evaluation_strategy: string
            Strategy to evaluate interaction potential.
        regularization_type: string
            Type of regularization to use for force law at separations below
            specified separation (constant_extrapolation, linear_extrapolation).
        regularization_separation: float
            Use specified regularization type for separations smaller than
            this value.
        integration_segments: int
            Number of integration segments to be used per beam element.
        gauss_points: int
            Number of Gauss points to be used per integration segment.
        potential_reduction_length: float
            Potential is smoothly decreased within this length when using the
            single length specific (SBIP) approach to enable an axial pull off
            force.
        automatic_differentiation: bool
            Use automatic differentiation via FAD.
        choice_master_slave: string
            Rule how to assign the role of master and slave to beam elements (if
            applicable) (lower_eleGID_is_slave, higher_eleGID_is_slave).
        """

        settings = {
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

        if regularization_type is not None:
            settings = settings | {
                "regularization": {
                    "type": regularization_type,
                    "separation": regularization_separation,
                }
            }

        if choice_master_slave is not None:
            settings = settings | {"choice_master_slave": choice_master_slave}

        # check if the section already exists, so one can either create the settings or runtime output settings first
        if "beam_potential" in self.input_file.sections:
            existing_entries = self.input_file.pop("beam_potential")
            existing_entries.update(settings)
            self.input_file["beam_potential"] = existing_entries
        else:
            self.input_file.add({"beam_potential": settings})

    def add_runtime_output(
        self,
        *,
        output_beam_potential=True,
        interval_steps=1,
        every_iteration=False,
        forces=True,
        moments=True,
        uids=True,
        per_ele_pair=True,
    ):
        """Set the basic runtime output options for beam potential
        interactions.

        Args
        ----
        output_beam_potential: bool
            If the output for beam potential should be written.
        interval_steps: int
            Interval at which output is written.
        every_iteration: bool
            If output at every Newton iteration should be written.
        forces: bool
            If the forces should be written.
        moments: bool
            If the moments should be written.
        uids: bool
            If the unique ids should be written.
        per_ele_pair: bool
            If the forces/moments should be written per element pair.
        """

        runtime_output_settings = {
            "runtime_output": {
                "interval_steps": interval_steps,
                "force": forces,
                "moment": moments,
                "every_iteration": every_iteration,
                "write_force_moment_per_elementpair": per_ele_pair,
                "write_uids": uids,
            }
        }

        if not output_beam_potential:
            runtime_output_settings["runtime_output"]["interval_steps"] = None

        # check if the section already exists, so one can either create the settings or runtime output settings first
        if "beam_potential" in self.input_file.sections:
            existing_entries = self.input_file.pop("beam_potential")
            existing_entries.update(runtime_output_settings)
            self.input_file["beam_potential"] = existing_entries
        else:
            self.input_file.add({"beam_potential": runtime_output_settings})

    def add_potential_charge_condition(self, *, geometry_set=None):
        """Add potential charge condition to geometry.

        Args
        ----
        geometry_set:
            Add potential charge condition to this set.
        """

        for i, (line_charge, func) in enumerate(
            zip(
                self.pot_law_line_charge_density, self.pot_law_line_charge_density_funcs
            )
        ):
            if func:
                self.mesh.add(func)

            bc = _BoundaryCondition(
                geometry_set,
                {"POTLAW": i + 1, "VAL": line_charge, "FUNCT": func},
                bc_type="DESIGN LINE BEAM POTENTIAL CHARGE CONDITIONS",
            )

            self.mesh.add(bc)
