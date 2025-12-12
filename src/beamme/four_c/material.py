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
"""This file implements materials for 4C beams and solids."""

import numpy as _np

from beamme.core.material import MaterialBeamBase as _MaterialBeamBase
from beamme.core.material import MaterialSolidBase as _MaterialSolidBase


class MaterialReissner(_MaterialBeamBase):
    """Holds material definition for Reissner beams."""

    def __init__(
        self,
        shear_correction=1.0,
        *,
        by_modes=False,
        scale_axial_rigidity=1.0,
        scale_shear_rigidity=1.0,
        scale_torsional_rigidity=1.0,
        scale_bending_rigidity=1.0,
        **kwargs,
    ):
        if by_modes:
            mat_string = "MAT_BeamReissnerElastHyper_ByModes"
        else:
            mat_string = "MAT_BeamReissnerElastHyper"

        super().__init__(material_string=mat_string, **kwargs)

        # Shear factor for Reissner beam.
        self.shear_correction = shear_correction

        self.by_modes = by_modes

        # Scaling factors to influence a single stiffness independently
        self.scale_axial_rigidity = scale_axial_rigidity
        self.scale_shear_rigidity = scale_shear_rigidity
        self.scale_torsional_rigidity = scale_torsional_rigidity
        self.scale_bending_rigidity = scale_bending_rigidity

        if not by_modes and not all(
            _np.isclose(x, 1.0)
            for x in (
                scale_axial_rigidity,
                scale_shear_rigidity,
                scale_torsional_rigidity,
                scale_bending_rigidity,
            )
        ):
            raise ValueError(
                "Scaling factors are only supported for MAT_BeamReissnerElastHyper_ByModes"
            )

    def dump_to_list(self):
        """Return a list with the (single) item representing this material."""

        if self.radius is None or self.youngs_modulus is None:
            raise ValueError(
                "Radius and Young's modulus must be provided for beam materials."
            )

        if (
            self.area is None
            and self.mom2 is None
            and self.mom3 is None
            and self.polar is None
        ):
            area, mom2, mom3, polar = self.calc_area_stiffness()
        elif (
            self.area is not None
            and self.mom2 is not None
            and self.mom3 is not None
            and self.polar is not None
        ):
            area = self.area
            mom2 = self.mom2
            mom3 = self.mom3
            polar = self.polar
        else:
            raise ValueError(
                "Either all relevant material parameters are set "
                "by the user, or a circular cross-section will be assumed. "
                "A combination is not possible"
            )

        if self.by_modes:
            shear_modulus = self.youngs_modulus / (2.0 * (1.0 + self.nu))

            data = {
                "EA": (self.youngs_modulus * area) * self.scale_axial_rigidity,
                "GA2": (shear_modulus * area * self.shear_correction)
                * self.scale_shear_rigidity,
                "GA3": (shear_modulus * area * self.shear_correction)
                * self.scale_shear_rigidity,
                "GI_T": (shear_modulus * polar) * self.scale_torsional_rigidity,
                "EI2": (self.youngs_modulus * mom2) * self.scale_bending_rigidity,
                "EI3": (self.youngs_modulus * mom3) * self.scale_bending_rigidity,
                "RhoA": self.density * area,
                "MASSMOMINPOL": self.density * (mom2 + mom3),
                "MASSMOMIN2": self.density * mom2,
                "MASSMOMIN3": self.density * mom3,
            }

        else:
            data = {
                "YOUNG": self.youngs_modulus,
                "POISSONRATIO": self.nu,
                "DENS": self.density,
                "CROSSAREA": area,
                "SHEARCORR": self.shear_correction,
                "MOMINPOL": polar,
                "MOMIN2": mom2,
                "MOMIN3": mom3,
            }

        if self.interaction_radius is not None:
            data["INTERACTIONRADIUS"] = self.interaction_radius

        return {"MAT": self.i_global + 1, self.material_string: data}


class MaterialReissnerElastoplastic(MaterialReissner):
    """Holds elasto-plastic material definition for Reissner beams."""

    def __init__(
        self,
        *,
        yield_moment=None,
        isohardening_modulus_moment=None,
        torsion_plasticity=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.material_string = "MAT_BeamReissnerElastPlastic"

        if yield_moment is None or isohardening_modulus_moment is None:
            raise ValueError(
                "The yield moment and the isohardening modulus for moments must be specified "
                "for plasticity."
            )

        self.yield_moment = yield_moment
        self.isohardening_modulus_moment = isohardening_modulus_moment
        self.torsion_plasticity = torsion_plasticity

    def dump_to_list(self):
        """Return a list with the (single) item representing this material."""
        super_list = super().dump_to_list()
        mat_dict = super_list[self.material_string]
        mat_dict["YIELDM"] = self.yield_moment
        mat_dict["ISOHARDM"] = self.isohardening_modulus_moment
        mat_dict["TORSIONPLAST"] = self.torsion_plasticity
        return super_list


class MaterialKirchhoff(_MaterialBeamBase):
    """Holds material definition for Kirchhoff beams."""

    def __init__(self, is_fad=False, **kwargs):
        super().__init__(material_string="MAT_BeamKirchhoffElastHyper", **kwargs)
        self.is_fad = is_fad

    def dump_to_list(self):
        """Return a list with the (single) item representing this material."""

        if self.radius is None or self.youngs_modulus is None:
            raise ValueError(
                "Radius and Young's modulus must be provided for beam materials."
            )

        if (
            self.area is None
            and self.mom2 is None
            and self.mom3 is None
            and self.polar is None
        ):
            area, mom2, mom3, polar = self.calc_area_stiffness()
        elif (
            self.area is not None
            and self.mom2 is not None
            and self.mom3 is not None
            and self.polar is not None
        ):
            area = self.area
            mom2 = self.mom2
            mom3 = self.mom3
            polar = self.polar
        else:
            raise ValueError(
                "Either all relevant material parameters are set "
                "by the user, or a circular cross-section will be assumed. "
                "A combination is not possible"
            )
        data = {
            "YOUNG": self.youngs_modulus,
            "SHEARMOD": self.youngs_modulus / (2.0 * (1.0 + self.nu)),
            "DENS": self.density,
            "CROSSAREA": area,
            "MOMINPOL": polar,
            "MOMIN2": mom2,
            "MOMIN3": mom3,
            "FAD": self.is_fad,
        }
        if self.interaction_radius is not None:
            data["INTERACTIONRADIUS"] = self.interaction_radius
        return {"MAT": self.i_global + 1, self.material_string: data}


class MaterialEulerBernoulli(_MaterialBeamBase):
    """Holds material definition for Euler Bernoulli beams."""

    def __init__(self, **kwargs):
        super().__init__(
            material_string="MAT_BeamKirchhoffTorsionFreeElastHyper", **kwargs
        )

    def dump_to_list(self):
        """Return a list with the (single) item representing this material."""

        if self.radius is None or self.youngs_modulus is None:
            raise ValueError(
                "Radius and Young's modulus must be provided for beam materials."
            )

        area, mom2, _, _ = self.calc_area_stiffness()
        if self.area is None and self.mom2 is None:
            area, mom2, _, _ = self.calc_area_stiffness()
        elif self.area is not None and self.mom2 is not None:
            area = self.area
            mom2 = self.mom2
        else:
            raise ValueError(
                "Either all relevant material parameters are set "
                "by the user, or a circular cross-section will be assumed. "
                "A combination is not possible"
            )
        data = {
            "YOUNG": self.youngs_modulus,
            "DENS": self.density,
            "CROSSAREA": area,
            "MOMIN": mom2,
        }
        return {"MAT": self.i_global + 1, self.material_string: data}


class MaterialSolid(_MaterialSolidBase):
    """Base class for a material for solids."""

    def __init__(self, material_string=None, **kwargs):
        """Set the material values for a solid."""
        self.material_string = material_string
        super().__init__(**kwargs)

    def dump_to_list(self):
        """Return a list with the (single) item representing this material."""

        return {"MAT": self.i_global + 1, self.material_string: self.data}

    def get_all_contained_materials(self) -> list["MaterialSolid"]:
        """Get all sub materials contained in this material, also nested ones.

        Returns:
            All MaterialSolid objects contained in this material, including those
            found recursively in nested `MATIDS` fields, in depth-first traversal
            order.
        """
        contained_materials = []
        if "MATIDS" in self.data:
            for item in self.data["MATIDS"]:
                if isinstance(item, MaterialSolid):
                    contained_materials.append(item)
                    contained_materials.extend(item.get_all_contained_materials())
        return contained_materials


class MaterialStVenantKirchhoff(MaterialSolid):
    """Holds material definition for StVenant Kirchhoff solids."""

    def __init__(self, youngs_modulus=None, nu=None, density=None):
        if youngs_modulus is None or nu is None:
            raise ValueError(
                "Young's modulus and Poisson's ratio must be provided "
                "for StVenant Kirchhoff solid materials."
            )
        data = {"YOUNG": youngs_modulus, "NUE": nu}
        if density is not None:
            data["DENS"] = density
        super().__init__(
            material_string="MAT_Struct_StVenantKirchhoff",
            data=data,
        )
