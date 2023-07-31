# PDE-Solvers
A set of numerical solvers for partial differential equations in 2D with boundary conditions on arbitrarily shaped domains.

pde_solver.py is the main script containing all the relevant functions, including solve_pde(), as well as object classes
for creating binary matrices of Rectangles, superellipses, circles, etc which can be used to describe the domain(s)
of interest. It also contains gridder(), a function which allows the use of non-uniform grids.
