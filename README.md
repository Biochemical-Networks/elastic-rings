# Elastic Rings Solver

A program for solving geometrically non-linear elastic rings with the deal.II finite element method (FEM) library.

This program was written to test whether beam theory was a valid assumption for bending of actin filaments that form rings.
Results from this program were used in the work presented in x.
However, the code developed is generally applicable to any problem where a geometrically nonlinear model of elasticity is applied to an object with rectangular cuboid geometry with either essential Dirichlet boundary conditions or linear constraints applied to paired boundary faces.
The code may also be used as a starting point to implement other nonlinear models of elasticity, more complex geometries, or other types of boundary conditions, although it is strongly recommended to also look through the rich library of [tutorials that are provided as part of the deal.II documentation](https://www.dealii.org/current/doxygen/deal.II/Tutorial.html).
Implementation of other models of elasticity is made relatively easy here by the use of the automatic differentiation capability of the deal.II library, which here calls wrappers for the Trilinos Sacado package.
The use of linear constraints on pairs of boundary faces, in place of essential Dirichlet boundary conditions, is similar to what are known as [tie constraints in the proprietary software Abaqus](https://abaqus-docs.mit.edu/2017/English/SIMACAECAERefMap/simacae-t-itnhelptied.htm).
These constraints allow one to solve for situations where one part of the surface of the object has been glued to another part of its surface, but one does not want to constrain the surfaces to some pre-defined position and orientation.

This program uses Newton's method to iteratively solve for the displacement vector field.
This method requires a relatively good guess of the final solution.
If this is not able to be given, it can help to solve a series of intermediate problems.
Here, it is possible to define multiple stages with different boundary conditions, where within each stage, the boundary is iteratively moved from the previous to the next through a simple linear combination.
It is also possible to begin by solving with a coarse mesh, and once an approximate solution has been found, run a number of mesh refinements (albeit non-adaptive).

The parameters and definitions for the mesh, elastic model, linear and nonlinear solver, boundary domains, boundary conditions, and I/O are specified in a configuration file.

## Installation

The program depends on the deal.II library, which itself has many dependencies, although many are not needed for its use here.
The version of deal.II that this code was developed with is 9.3.0.
The [installation documentation for deal.II is extensive](https://www.dealii.org/current/readme.html) and as of the time of this writing, [the mailing list](https://groups.google.com/g/dealii) is active and has members willing to give advice on installation issues.
The optional dependencies of deal.II required here are BLAS/LAPACK, GSL, muparser, Trilinos, UMFPACK.
Trilinos itself is a large library with many dependencies; advice on building it for deal.II can be found [here](https://www.dealii.org/current/external-libs/trilinos.html).
When building Trilinos and deal.II for this program, [it was discovered](https://github.com/dealii/dealii/pull/12424/commits) that they should both be built with MPI, contrary to the documentation, and should use the same version of the preferred MPI implementation, although this program has not be designed to run parallel calculations.

To build the program starting from the main directory of the repository, run
```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=[desired install location]
```
The option `CMAKE_BUILD_TYPE` can also be set to `Debug`.
The deal.II library location should be detected by cmake, but if not, it can be specified with the `deal.II_DIR` option.

## Running a calculation

To run a calculation, a configuration file is required.
Two example configuration files are provided in the `examples` directory.
One, `beam.conf` takes a long beam lying along the X-axis and applies a Dirichlet boundary condition such that one end is shifted up in the Y-axis, but without changing the angle of that face.
The other, `ring.conf`, also starts with a beam, but applies pair constraints to glue the ends of the beam together to form a ring (it does so by making a very good initial guess of the initial solution).
To run the calculation with the configuration file called `beam.conf`, run
```
solve-ring-nonlinear examples/beam.conf
```

The configuration file format is defined by the deal.II class [ParameterHandler](https://www.dealii.org/current/doxygen/deal.II/classParameterHandler.html).
The format is, for the most part, key value pairs delimited by an equals sign, where these pairs may be organized into subsections.
A subsection line is preceded by `subsection`, while a key value pair line is preceded by `set`.
To see a description of all subsections and keys, along with any default values, run
```
solve-ring-nonlinear -h
```

Three of the subsections can have multiple version.
These are `Boundary domain`, `Boundary condition`, and `Boundary stage`.
They are indexed from 0, and up to 10 can be defined (the code can be easily be modified to allow for more).
The boundary conditions reference the boundary domains, and specify the type of boundary condition to be applied to the domains.
For a Dirichlet boundary condition, only `Associated domain` must be set to the domain index it will be applied to.
For a pair constraint condition, both `Anchor domain` and `Constrained domain` must be set.
For each boundary stage, a subsection for all defined boundary conditions must be defined, with the function (either constants for the Dirichlet condition, or a linear equation for the pair constraint).
For the pair constraint function defined in a boundary stage, the x variable is the constrained boundary minus the anchor boundary, where the position vectors of the boundary are in the material reference.
The y and z variables are the material position vectors of the anchor domain.
Currently if a pair constraint is used, no other boundary conditions can be used (or at least it must be index 0 and only Dirichlet conditions follow; extending this would not be difficult).
At least two stages must be defined, as the general idea is to take steps from one stage to another.
For stages other than 0, the number of increments should be set.

In addition to data for analysis, the program will also output archive files that can be read back in to restart the simulation after some number of increments, stages, and refinements.
The checkpoint format is to append `_[stage index]-[refinement index]-[gamma]` to the given output filename prefix.
To run a restart calculation, `Starting refinement`, `Input checkpoint`, and `Starting stage` should be specified.
The program will not however, know how many increments have already been run.

## Analysis and visualization

The program will integrated values of the traction forces on the boundaries (currently assuming that boundary 0 and boundary 1 correspond to left and right, respectively) to standard out when it has completed.
These are also printed to a file `[Output filename prefix]_integrated_[checkpoint].dat`.
See above section for checkpoint formatting.
The displacement vector field, material and spatial normal vector field, and the material and spatial stress vector field is also output in vtu format as `[Output filename prefix]_faces_[checkpoint].vtu`, which is viewable with [Paraview](https://www.paraview.org/).
In Paraview, apply the "Warp by Vector" filter to the displacement vectors to see the deformed object, and the "Glyph" filter to the stress vectors (either to the undeformed object for the material stress vector or the deformed object for the spatial stress vector) to see the stress .
The norm of the gradient of the displacement vector field is also included (a scalar field), which can be used to determine whether and where a linear assumption to the strain is valid (i.e. where the magnitude of the strain is much less than 1) by viewing it with a colour map.

## References

[deal.II main page](https://www.dealii.org/)

[Paraview main page](https://www.paraview.org/)

Some textbooks and material to understand continuum mechanics, elasticity, and FEM:

[Wolfgang Bangerth's online lecture course](https://www.math.colostate.edu/~bangerth/videos.html), highly recommended, he is one of the core deal.II developers, free

[An Introduction to Continuum Mechanics, J.N. Reddy](https://www.cambridge.org/nl/academic/subjects/engineering/solid-mechanics-and-materials/introduction-continuum-mechanics-2nd-edition?format=HB), not free

[Principles of Continuum Mechanics, J.N. Reddy](https://www.cambridge.org/nl/academic/subjects/engineering/solid-mechanics-and-materials/principles-continuum-mechanics-conservation-and-balance-laws-applications-2nd-edition?format=HB&isbn=9781107199200), simplified version of the above, not free

[Continuum Mechanics and Thermodynamics, Ellad B. Tadmor, Ronald E. Miller, Ryan S. Elliott](https://www.cambridge.org/nl/academic/subjects/physics/mathematical-methods/continuum-mechanics-and-thermodynamics-fundamental-concepts-governing-equations?format=HB), not free

[Introduction to Numerical Methods for Variational Problems, Hans Petter Langtangen, Kent-Andre Mardal](https://github.com/hplgit/fem-book), free

[Variational Methods with Applications in Science and Engineering, Kevin W. Cassel](https://www.cambridge.org/nl/academic/subjects/engineering/engineering-mathematics-and-programming/variational-methods-applications-science-and-engineering?format=HB), not free
