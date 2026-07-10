# %% [markdown]
# $$
# % Define TeX macros for this document
# \def\vv#1{\boldsymbol{#1}}
# $$
#
# # Example 2: Basic mesh generation functions
#
# This example shall showcase the core mesh generation functions provided in BeamMe.

# %% [markdown]
# In a first step we define the type of beam elements we want to use in this example.
# The mesh generation functions in BeamMe are agnostic with respect to the employed beam formulation, i.e., every possible beam type can be used with every mesh generation function.
# For this example we use the `Beam3rLine2Line2` class, which represents a two-noded beam element.
# Hand in hand with the beam type goes the beam material that stores information about the beam-cross section.
# In this example we mainly use it for defining the radius of the beams for visualization purposes.

# %%
from beamme.four_c.element_beam import Beam3rLine2Line2
from beamme.four_c.material import MaterialReissner

beam_type = Beam3rLine2Line2
beam_mat = MaterialReissner

# %% [markdown]
# The `Mesh` class is the core BeamMe class that will hold all the nodes, elements, materials, and geometry sets for the created geometries.

# %%
from beamme.core.mesh import Mesh

# %% [markdown]
# ## Straight lines
#
# We already have everything we need to create basic geometries.
# Let's start of with the most basic one, a straight line.
# For that we need the `create_beam_mesh_line` function.
# Mesh generation functions always require to provide a mesh to add the created geometry to, a beam type and a beam material.
# In this example we create a line between the points $\vv{p} = [0,0,0]$ and $\vv{q}=[1,0,0]$ with 3 equally spaced beam elements:
# %%
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line

mesh = Mesh()
material = beam_mat(radius=0.01, youngs_modulus=1.0, nu=0.3, density=1.0)
create_beam_mesh_line(mesh, beam_type, material, [0, 0, 0], [1, 0, 0], n_el=3)

# %% [markdown]
# We can see that the mesh creation function returned a dictionary containing `GeometrySet`s.
# These geometry sets can be used to define boundary conditions.
# In this case we get the following sets:
# - `start`: A geometry set referring to the start node of the line
# - `end`: A geometry set referring to the end node of the line
# - `line`: A geometry set referring to all created beam elements along the line
#
# Later we will dive closer into `GeometrySets`.
#
# We can directly have a look at the created geometry with the `Mesh.display_pyvista` method:

# %%
mesh.display_pyvista()

# %% [markdown]
# TODO: describe vtu output

# %% [markdown]
# ## TODO
