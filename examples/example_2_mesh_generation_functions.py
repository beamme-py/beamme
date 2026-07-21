# %% [markdown]
# $$
# % Define TeX macros for this document
# \def\vv#1{\boldsymbol{#1}}
# $$
#
# # Mesh creation functions
# The mesh generation procedure in BeamMe consists of two main functionalities: _mesh creation functions_ and _mesh manipulation functions_. Combining these two functionalities allows the user to create arbitrary complex one dimensional meshes. In this example, we will focus on the main mesh creation functions in BeamMe:
# - **Straight line**
# - **Circle and arc**
# - **Helix**
# - **Parametric curve**
# - **NURBS curve**
#
# Besides the geometry parameters, a few additional parameters are required for the mesh creation functions. These are:
# - A container of type `Mesh` that will hold all the nodes, elements, materials, and geometry sets for the created geometries.
# - The type of the elements to be created. Among other parameters, this defines the number of nodes per created element. The mesh generation functions in BeamMe are agnostic with respect to the employed beam formulation, i.e., every possible beam type can be used with every mesh generation function. Some of the available beam types in BeamMe are:
#     - `Beam2`: A generic two noded beam element.
#     - `Beam3`: A generic three noded beam element.
#     - `Beam4`: A generic four noded beam element.
# - A material of type `MaterialBeamBase` to be assigned to the created elements.
# - To define the size and number of created beam elements, one of the following parameters is required:
#     - `n_el`: Number of elements for the created curve.
#     - `l_el`: Desired length of beam elements.
#     - `node_positions_of_elements`: A list of lists of node positions for each element. This allows for non-uniformly sized elements along a curve.

# %% [markdown]
# ## Straight line
#
# A straight line between two points in space can be generated with the `create_beam_mesh_line` function.
#
# The following code creates a straight line between the points $\vv{p}_1 = [0, 0, 0]$ and $\vv{p}_2 = [2, -1, 1]$ with 3 elements of type `Beam3` and a radius of 0.05. Note that the radius is only relevant for visualization purposes.

# %%
from beamme.core.element_beam import Beam3
from beamme.core.material import MaterialBeamBase
from beamme.core.mesh import Mesh
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line

mesh = Mesh()
material = MaterialBeamBase(radius=0.05)

create_beam_mesh_line(mesh, Beam3, material, [0, 0, 0], [2, -1, 1], n_el=3)

mesh.display_pyvista()

# %% [markdown]
# ## Circle and arc
#
# Circular arcs can be created with `create_beam_mesh_arc_segment_via_axis`. The arc is defined by an axis of rotation, a point on this axis, a start point, and an angle. The start point determines the radius of the arc with respect to the rotation axis.
#
# The following example creates a quarter circle in the $xy$-plane. The axis of rotation is the $z$-axis, the center lies at the origin, and the arc starts at $\vv{p}_1 = [1, 0, 0]$.

# %%
import numpy as np

from beamme.core.element_beam import Beam3
from beamme.core.material import MaterialBeamBase
from beamme.core.mesh import Mesh
from beamme.mesh_creation_functions.beam_arc import (
    create_beam_mesh_arc_segment_via_axis,
)

mesh = Mesh()
material = MaterialBeamBase(radius=0.05)

create_beam_mesh_arc_segment_via_axis(
    mesh,
    Beam3,
    material,
    axis=[0, 0, 1],
    axis_point=[0, 0, 0],
    start_point=[1, 0, 0],
    angle=0.5 * np.pi,
    n_el=3,
)

mesh.display_pyvista()

# %% [markdown]
# ## Helix
#
# A helix can be generated with `create_beam_mesh_helix`. Similar to the arc creation function, the helix is defined by an axis, a point on the axis, and a start point. The shape of the helix is then determined by providing additional parameters: the helix angle, the total height, and the number of turns. Since these parameters are not independent, only two of the three need to be specified.
#
# The following example creates a helix around the $z$-axis with radius 1, height 2, and 3 full turns.

# %%
from beamme.core.element_beam import Beam3
from beamme.core.material import MaterialBeamBase
from beamme.core.mesh import Mesh
from beamme.mesh_creation_functions.beam_helix import create_beam_mesh_helix

mesh = Mesh()
material = MaterialBeamBase(radius=0.05)

create_beam_mesh_helix(
    mesh,
    Beam3,
    material,
    axis_vector=[0, 0, 1],
    axis_point=[0, 0, 0],
    start_point=[1, 0, 0],
    height_helix=2,
    turns=3,
    n_el=20,
)

mesh.display_pyvista()

# %% [markdown]
# ## Parametric curve
#
# For more general geometries, BeamMe can create beam meshes from user-defined parametric curves. The curve is given as a function
#
# $$
# \vv{r}(t) = [x(t), y(t), z(t)].
# $$
#
# together with a parameter interval. BeamMe evaluates this function and creates beam elements along the resulting curve. BeamMe automatically ensures that the created elements are equally spaced. For three-dimensional curves, BeamMe automatically generates a suitable triad (rotation) field along the curve using smallest rotation mappings. Alternatively, the user can provide a custom triad field if a specific beam orientation is required.
#
# The following example creates a curve defined by
#
# $$
# \vv{r}(t) = [t, 0, e^{-t^2}] \quad \text{with} \quad t \in [-2, 2].
# $$
#
# Since derivatives of the parametric curve are required internally, the curve should be implemented using `autograd.numpy` instead of standard `numpy`.

# %%
import autograd.numpy as np_ad

from beamme.core.element_beam import Beam3
from beamme.core.material import MaterialBeamBase
from beamme.core.mesh import Mesh
from beamme.mesh_creation_functions.beam_parametric_curve import (
    create_beam_mesh_parametric_curve,
)


def parametric_curve(t):
    x = t
    y = 0
    z = np_ad.exp(-(t**2))
    return np_ad.array([x, y, z])


mesh = Mesh()
material = MaterialBeamBase(radius=0.05)

create_beam_mesh_parametric_curve(
    mesh,
    Beam3,
    material,
    parametric_curve,
    (-2, 2),
    n_el=10,
)

mesh.display_pyvista()

# %% [markdown]
# ## NURBS curve
#
# BeamMe also supports mesh creation from spline curves provided by [splinepy](https://github.com/isosuite/splinepy). This is useful when geometries are already available as B-spline or NURBS curves, for example from CAD-like workflows.
#
# The following example defines a quadratic NURBS curve using its degree, knot vector, control points, and weights. The curve is then discretized into beam elements using `create_beam_mesh_from_splinepy`.

# %%
import splinepy

from beamme.core.element_beam import Beam3
from beamme.core.material import MaterialBeamBase
from beamme.core.mesh import Mesh
from beamme.mesh_creation_functions.beam_splinepy import (
    create_beam_mesh_from_splinepy,
)

mesh = Mesh()
material = MaterialBeamBase(radius=0.05)

nurbs_curve = splinepy.NURBS(
    degrees=[2],
    knot_vectors=[[0, 0, 0, 0.5, 1, 1, 1]],
    control_points=[[0, 0, 0], [1, -1, 2], [1, 1, 2], [2, 0, 0]],
    weights=[[1.0], [1.0], [1.0], [1.0]],
)

create_beam_mesh_from_splinepy(
    mesh,
    Beam3,
    material,
    nurbs_curve,
    n_el=10,
)

mesh.display_pyvista()

# %%
