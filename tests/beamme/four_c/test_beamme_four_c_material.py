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
"""This file unit tests materials for 4C."""

from beamme.four_c.material import (
    MaterialKirchhoff,
    MaterialReissner,
    MaterialReissnerElastoplastic,
    MaterialSolid,
    MaterialStVenantKirchhoff,
)


def test_beamme_four_c_material_reissner(assert_results_close):
    """Test Reissner material."""

    mat = MaterialReissner(
        radius=0.5,
        youngs_modulus=1234.56,
        nu=0.33,
        density=0.123,
        interaction_radius=2.4,
        shear_correction=17.15,
    )

    # set manually
    mat.i_global = 0

    mat_expected = {
        "MAT": 1,
        "MAT_BeamReissnerElastHyper": {
            "YOUNG": 1234.56,
            "POISSONRATIO": 0.33,
            "DENS": 0.123,
            "CROSSAREA": 0.7853981633974483,
            "SHEARCORR": 17.15,
            "MOMINPOL": 0.09817477042468103,
            "MOMIN2": 0.04908738521234052,
            "MOMIN3": 0.04908738521234052,
            "INTERACTIONRADIUS": 2.4,
        },
    }

    assert_results_close(mat.dump_to_list(), mat_expected)


def test_beamme_four_c_material_reissner_by_modes(assert_results_close):
    """Test Reissner material by modes with scaling factors."""

    mat = MaterialReissner(
        radius=0.5,
        youngs_modulus=1234.56,
        nu=0.33,
        density=0.123,
        interaction_radius=2.4,
        shear_correction=17.15,
        by_modes=True,
        scale_axial_rigidity=1.1,
        scale_shear_rigidity=1.2,
        scale_torsional_rigidity=1.3,
        scale_bending_rigidity=1.4,
    )

    # set manually
    mat.i_global = 0

    mat_expected = {
        "MAT": 1,
        "MAT_BeamReissnerElastHyper_ByModes": {
            "EA": 1066.5832722643493,
            "GA2": 7501.8057905674295,
            "GA3": 7501.8057905674295,
            "GI_T": 59.234375168474614,
            "EI2": 84.84185120284594,
            "EI3": 84.84185120284594,
            "RhoA": 0.09660397409788614,
            "MASSMOMINPOL": 0.012075496762235767,
            "MASSMOMIN2": 0.006037748381117884,
            "MASSMOMIN3": 0.006037748381117884,
            "INTERACTIONRADIUS": 2.4,
        },
    }

    assert_results_close(mat.dump_to_list(), mat_expected)


def test_beamme_four_c_material_reissner_elasto_plastic(assert_results_close):
    """Test the elasto plastic Reissner beam material."""

    kwargs = {
        "radius": 0.1,
        "nu": 1.0,
        "density": 1.0,
        "youngs_modulus": 1000,
        "interaction_radius": 2.0,
        "shear_correction": 5.0 / 6.0,
        "yield_moment": 2.3,
        "isohardening_modulus_moment": 4.5,
        "torsion_plasticity": False,
    }

    ref_dict = {
        "MAT": 69,
        "MAT_BeamReissnerElastPlastic": {
            "YOUNG": 1000,
            "POISSONRATIO": 1.0,
            "DENS": 1.0,
            "CROSSAREA": 0.031415926535897934,
            "SHEARCORR": 0.8333333333333334,
            "MOMINPOL": 0.00015707963267948968,
            "MOMIN2": 7.853981633974484e-05,
            "MOMIN3": 7.853981633974484e-05,
            "INTERACTIONRADIUS": 2.0,
            "YIELDM": 2.3,
            "ISOHARDM": 4.5,
            "TORSIONPLAST": False,
        },
    }

    mat = MaterialReissnerElastoplastic(**kwargs)
    mat.i_global = 68

    assert_results_close(mat.dump_to_list(), ref_dict)

    ref_dict["MAT_BeamReissnerElastPlastic"]["TORSIONPLAST"] = True
    kwargs["torsion_plasticity"] = True
    mat = MaterialReissnerElastoplastic(**kwargs)
    mat.i_global = 68
    assert_results_close(mat.dump_to_list(), ref_dict)


def test_beamme_four_c_material_kirchhoff_material(assert_results_close):
    """Test the Kirchhoff Love beam material."""

    def set_stiff(material):
        """Set the material properties for the beam material."""
        material.area = 2.0
        material.mom2 = 3.0
        material.mom3 = 4.0
        material.polar = 5.0

    material = MaterialKirchhoff(
        youngs_modulus=1000, radius=1.0, nu=1.0, density=1.0, is_fad=True
    )
    material.i_global = 26
    set_stiff(material)
    assert_results_close(
        material.dump_to_list(),
        {
            "MAT": 27,
            "MAT_BeamKirchhoffElastHyper": {
                "YOUNG": 1000,
                "SHEARMOD": 250.0,
                "DENS": 1.0,
                "CROSSAREA": 2.0,
                "MOMINPOL": 5.0,
                "MOMIN2": 3.0,
                "MOMIN3": 4.0,
                "FAD": True,
            },
        },
    )

    material = MaterialKirchhoff(
        youngs_modulus=1000, radius=1.0, nu=1.0, density=1.0, is_fad=False
    )
    material.i_global = 26
    set_stiff(material)
    assert_results_close(
        material.dump_to_list(),
        {
            "MAT": 27,
            "MAT_BeamKirchhoffElastHyper": {
                "YOUNG": 1000,
                "SHEARMOD": 250.0,
                "DENS": 1.0,
                "CROSSAREA": 2.0,
                "MOMINPOL": 5.0,
                "MOMIN2": 3.0,
                "MOMIN3": 4.0,
                "FAD": False,
            },
        },
    )

    material = MaterialKirchhoff(
        youngs_modulus=1000, radius=1.0, nu=1.0, density=1.0, interaction_radius=1.1
    )
    material.i_global = 26
    set_stiff(material)
    assert_results_close(
        material.dump_to_list(),
        {
            "MAT": 27,
            "MAT_BeamKirchhoffElastHyper": {
                "YOUNG": 1000,
                "SHEARMOD": 250.0,
                "DENS": 1.0,
                "CROSSAREA": 2.0,
                "MOMINPOL": 5.0,
                "MOMIN2": 3.0,
                "MOMIN3": 4.0,
                "FAD": False,
                "INTERACTIONRADIUS": 1.1,
            },
        },
    )


def test_beamme_four_c_material_stvenantkirchhoff_solid(assert_results_close):
    """Test that the solid with St_Venant Kirchhoff material."""

    material = MaterialStVenantKirchhoff(youngs_modulus=157, nu=0.17, density=6.1e-7)
    material.i_global = 3
    assert_results_close(
        material.dump_to_list(),
        {
            "MAT": 4,
            "MAT_Struct_StVenantKirchhoff": {
                "YOUNG": 157,
                "NUE": 0.17,
                "DENS": 6.1e-07,
            },
        },
    )


def test_beamme_four_c_material_sub_materials():
    """Test that sub-materials are correctly returned from the material."""

    material_1_1 = MaterialSolid(material_string="mat_1_1")
    material_1_2 = MaterialSolid(material_string="mat_1_2")
    material_1 = MaterialSolid(
        material_string="mat_1", data={"MATIDS": [material_1_1, material_1_2]}
    )

    material_2 = MaterialSolid(material_string="mat_2")

    material_3_1 = MaterialSolid(material_string="mat_3_1")
    material_3_2_1 = MaterialSolid(material_string="mat_3_2_1")
    material_3_2 = MaterialSolid(
        material_string="mat_3_2", data={"MATIDS": [material_3_2_1]}
    )
    material_3_3 = MaterialSolid(material_string="mat_3_3")
    material_3 = MaterialSolid(
        material_string="mat_3",
        data={"MATIDS": [material_3_1, material_3_2, material_3_3]},
    )

    material = MaterialSolid(
        material_string="mat", data={"MATIDS": [material_1, material_2, material_3]}
    )
    sub_materials = material.get_all_contained_materials()

    sub_materials_reference = [
        material_1,
        material_1_1,
        material_1_2,
        material_2,
        material_3,
        material_3_1,
        material_3_2,
        material_3_2_1,
        material_3_3,
    ]
    assert len(sub_materials_reference) == len(sub_materials)
    for mat_reference, mat_test in zip(sub_materials_reference, sub_materials):
        assert mat_reference is mat_test
