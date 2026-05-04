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
"""This script is used to test the header functions for 4C."""

import pytest
from fourcipp.fourc_input import FourCInput

from beamme.core.conf import bme
from beamme.four_c.header_functions import (
    set_beam_contact_runtime_output,
    set_beam_contact_section,
    set_beam_to_solid_meshtying,
    set_binning_strategy_section,
    set_header_static,
    set_runtime_output,
)
from beamme.four_c.input_file import InputFile


def test_integration_four_c_header_functions_static(
    get_corresponding_reference_file_path,
    assert_results_close,
):
    """Test the default static header function."""

    # Create input file.
    input_file = InputFile()

    # Set the header.
    set_header_static(input_file, time_step=0.1, n_steps=17, load_lin=True)
    set_runtime_output(input_file, output_triad=False)

    set_binning_strategy_section(
        input_file, binning_bounding_box=[1, 2, 3, 4, 5, 6], binning_cutoff_radius=0.69
    )

    set_beam_to_solid_meshtying(
        input_file,
        bme.bc.beam_to_solid_volume_meshtying,
        contact_discretization="mortar",
    )

    set_beam_to_solid_meshtying(
        input_file,
        bme.bc.beam_to_solid_surface_meshtying,
        contact_discretization="gp",
        coupling_type="consistent_fad",
        segmentation=False,
    )

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), input_file)


@pytest.mark.parametrize(
    "additional_identifier, time_step, n_steps, total_time",
    [
        ("all", 0.01, 17, 2.1),
        ("no_total_time", 0.01, 17, None),
        ("no_n_steps", 0.01, None, 2.1),
        ("no_time_step", None, 17, 2.1),
    ],
)
def test_integration_four_c_header_functions_static_time(
    get_corresponding_reference_file_path,
    assert_results_close,
    additional_identifier,
    time_step,
    n_steps,
    total_time,
):
    """Test the time setting options in the static header functions."""

    input_file = InputFile()

    set_header_static(
        input_file, time_step=time_step, n_steps=n_steps, total_time=total_time
    )

    assert_results_close(
        get_corresponding_reference_file_path(
            additional_identifier=additional_identifier
        ),
        input_file,
    )


def test_integration_four_c_header_functions_static_prestress(
    get_corresponding_reference_file_path,
    assert_results_close,
):
    """Test the static header function with non default prestressing
    parameter."""

    # Create input file.
    input_file = InputFile()

    # Set the header.
    set_header_static(
        input_file,
        time_step=0.1,
        n_steps=17,
        load_lin=True,
        prestress="mulf",
        prestress_time=1,
    )
    set_beam_to_solid_meshtying(
        input_file,
        bme.bc.beam_to_solid_volume_meshtying,
        contact_discretization="mortar",
        binning_parameters={
            "binning_bounding_box": [1, 2, 3, 4, 5, 6],
            "binning_cutoff_radius": 0.69,
        },
        couple_restart=True,
    )

    set_beam_to_solid_meshtying(
        input_file,
        bme.bc.beam_to_solid_surface_meshtying,
        contact_discretization="gp",
        segmentation=False,
        couple_restart=False,
    )

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), input_file)


def test_integration_four_c_header_functions_stress_output(
    get_corresponding_reference_file_path,
    assert_results_close,
):
    """Test the static header function with non default stress output
    parameter."""

    # Create input file.
    input_file = InputFile()

    # Set the header.
    set_header_static(
        input_file,
        time_step=0.1,
        n_steps=17,
        load_lin=True,
        write_stress="cauchy",
        write_strain="gl",
    )
    set_runtime_output(input_file, output_stress_strain=True)

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), input_file)


def test_integration_four_c_header_functions_beam_interaction(
    get_corresponding_reference_file_path, assert_results_close
):
    """Test the beam-to-beam contact header function with default parameter."""

    # Create input file.
    input_file = InputFile()

    # Add Beam contact section to file.
    set_beam_contact_section(
        input_file,
        binning_parameters={
            "binning_cutoff_radius": 5,
            "binning_bounding_box": [-1, -2, -3, 1, 2, 3],
        },
        beam_interaction_parameters={"repartition_strategy": "adaptive"},
    )

    # Add per default the runtime output.
    set_beam_contact_runtime_output(input_file)

    # Compare the output.
    assert_results_close(get_corresponding_reference_file_path(), input_file)


@pytest.mark.parametrize(
    ("create_nox_file", "nox_xml_file_kwarg", "xml_relative_path"),
    [
        (False, None, None),
        (True, None, "nonlinear_solver_parameters.nox.xml"),
        (True, "custom_name.xml", "custom_name.xml"),
    ],
)
def test_integration_four_c_header_functions_nonlinear_solver_parameters(
    get_corresponding_reference_file_path,
    assert_results_close,
    create_nox_file,
    nox_xml_file_kwarg,
    xml_relative_path,
    tmp_path,
):
    """Test that the NOX xml is exported correctly."""

    input_file = InputFile()
    set_header_static(
        input_file,
        total_time=1.0,
        n_steps=1,
        tol_increment=1e-4,
        tol_residuum=1e-5,
        max_iter=25,
        create_nox_file=create_nox_file,
    )
    temp_input_file_path = tmp_path / "nonlinear_solver_parameters.4C.yaml"
    input_file.dump(
        temp_input_file_path,
        nox_xml_file=nox_xml_file_kwarg,
        add_header_default=False,
        add_header_information=False,
        add_footer_application_script=False,
    )

    if create_nox_file:
        # Load the written input file and check the xml path in the input file
        written_input_file = FourCInput.from_4C_yaml(temp_input_file_path)
        assert (
            written_input_file["STRUCT NOX/Status Test"]["XML File"]
            == xml_relative_path
        )

        # Check the created xml
        assert_results_close(
            get_corresponding_reference_file_path(extension="xml"),
            tmp_path / xml_relative_path,
        )

    assert_results_close(
        get_corresponding_reference_file_path(
            additional_identifier=("with_nox" if create_nox_file else "")
            + ("_custom_name" if nox_xml_file_kwarg else "")
        ),
        tmp_path / "nonlinear_solver_parameters.4C.yaml",
    )
