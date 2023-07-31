# -*- coding: utf-8 -*-
"""
Uses pde_solver.py to solve Poissons's equation on a rectangular domain
with Dirichlet non-zero boundary conditions and a non-zero source.

Author: Oscar A. Nieves
"""

import numpy as np
import pde_solver as pde
import matplotlib.pyplot as plt

# Input parameters
Nmin = 21 # minimum points in grid along either direction
xlims = [0,1]
ylims = [0,1]
#x_res = [11,21,7]
x_res = [11]
y_res = x_res

[X, Y, x, y, dx, dy] = \
    pde.gridder(x_uniform = Nmin, 
                y_uniform = Nmin,
                x_dense_points = [0,0.5,0.7,1],
                y_dense_points = [0,0.5,0.7,1],
                display_grid = True,
                uniform = False,
                x_res = x_res, 
                y_res = y_res)

x0 = np.mean(xlims)
y0 = np.mean(ylims)
width = xlims[1]-xlims[0]
height = ylims[1]-ylims[0]

# Define solution domain and compute boundary
Omega = pde.Rectangle(X,Y,width,height,x0,y0)
domain = Omega.area
inner_domain = Omega.hole
boundary_list = [Omega.lower_boundary, Omega.upper_boundary,
                 Omega.left_boundary, Omega.right_boundary]

# Define boundary value functions for each boundary
BCs_on = 1
boundary_values_list = [BCs_on*np.sin(2*np.pi*X), BCs_on*np.sin(2*np.pi*X),
                        BCs_on*2*np.sin(2*np.pi*Y), BCs_on*2*np.sin(2*np.pi*Y)]

# Source function
f_source = 100*(X**2 + Y**2)
                       
# Left-hand side of the bvp in the form L*u = b where L is a linear operator
# containing derivatives and coefficient functions, u(x,y) is the solution vector
# to be determined and b is the source function (also callable and optional)
LHS = pde.Laplacian(x=x, y=y, dx=dx, dy=dy)

# Compute solution and plot
solution = pde.solve_pde(X = X, 
                         Y = Y, 
                         dx = dx, 
                         dy = dy, 
                         domain = domain, 
                         inner_domain = inner_domain,
                         boundary = boundary_list, 
                         boundary_values = boundary_values_list, 
                         LHS = LHS, 
                         source = f_source, 
                         plot_solution = True)

# Compute gradient field
stream = pde.Grad(x=x, y=y, dx=dx, dy=dy, f=solution)

# Plot vector field
pde.plot_stream(X=X, Y=Y, f=solution, stream=stream, color="black")