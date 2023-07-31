# -*- coding: utf-8 -*-
"""
%-----------------------------------------------------------------------------%
Description of the module:

A self-contained module (dependencies: NumPy, Numba, Scipy and Matplotlib) for 
solving 2-dimensional boundary-value problems associated with linear partial 
differential equations (PDEs) on simply OR multiply connected domains 
(e.g. containing holes).

It is originally intended for solving steady-state problems (in rectangular
coordinates only) which are expressible in the following general form:

    L*u(x,y) = b(x,y)

where L is some linear operator containing partial derivative and coefficient
functions, u(x,y) is the vector of solutions to be determined, and b(x,y) is 
an optional source term. In this way, we can solve equations such as:
    
    ∇^2 u(x,y) = 0 --> Laplace's equation
    ∇^2 u(x,y) = f(x,y) --> Poisson equation
    ∇^2 u(x,y) + k^2 u(x,y) = f(x,y) --> Helmholtz equation
    
It uses the finite difference method (FDM) with centered finite difference
schemes. Currently, the module contains functions to create 1st and 2nd order
partial derivative matrices, namely Dx, Dxx, Dy and Dyy, and the Laplacian
operator. However, the main function 'solve_pde()' takes in any left-hand
side operator LHS as an input, so as the user you can define this L matrix
using anything you like, even your own derivative matrices of arbitrary order.

There is an option to solve PDEs on a "non-uniform" grid in which the
step-sizes vary across the domain of solution. This is achieved via the function
"gridder()" which generates a non-uniform grid with respect to a list of
reference coordinates. For details, refer to the function's docstring. Please 
note that at the moment, gridder() uses linearly spaced points in between 
reference points.

For details on how to use the solver, refer to the script
"example_Poisson_rectangular_domain.py".
%-----------------------------------------------------------------------------%    
Author: Oscar A. Nieves
%-----------------------------------------------------------------------------%
"""
import numpy as np
import numba as nb
import scipy as sp
import time
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt

###############################################################################
# Geometry objects and Boundary points extraction
###############################################################################
class Shape:
    """A generic shape object consisting of 2 attributes (binary matrices):
        a filled area (of the shape) and the shape's boundary
    
    Args:
        area: a binary matrix with 1's in every point (x,y) that lies within
                the shape's domain
                
        boundary: a binary matrix with 1's in every point along the boundary 
                    enclosing the shape
    
    Functions:
        union: creates the union of two shapes: self and other, and then computes
                the binary matrix corresponding to the combined (overlapping)
                areas, the new boundary enclosing it and the hole enclosed
                within the boundary.
                
        insert_hole: inserts a hole of given shape into the current shape. It also
                     adds the inserted hole object to the self.subdomains list
                     which can be used for keeping track of all the holes and their
                     attributes (e.g. area, boundary). This function verifies if the
                     given hole shape lies entirely within the current Shape's area,
                     otherwise it will throw an error.
    """
    def __init__(self, 
                 area: np.ndarray = np.array([1]), 
                 boundary: np.ndarray = np.array([1]), 
        ): 
        if type(area) is not sp.sparse._csc.csc_matrix: area = csc_matrix(area)
        if type(boundary) is not sp.sparse._csc.csc_matrix: boundary = csc_matrix(boundary)
        self.area = area
        self.boundary = boundary
        self.hole = csc_matrix( np.logical_xor(area.toarray(), boundary.toarray()) )
        self.subdomains = []
        self.outer_boundary = self.extract_outer_boundary()
    
    def union(self, other):
        self_area = self.area.toarray()
        other_area = other.area.toarray()
        new_area = np.logical_or(self_area, other_area)
        new_boundary = extract_boundary(new_area)
        self.area = csc_matrix(new_area)
        self.boundary = csc_matrix(new_boundary)
        self.outer_boundary = self.extract_outer_boundary()
        self.hole = csc_matrix( np.logical_xor(self.area.toarray(), 
                                               self.boundary.toarray()) )
        
    def difference(self, other):
        self_area = self.area.toarray()
        other_area = other.area.toarray()
        intersection = np.logical_and(self_area, other_area)
        new_area = np.logical_and(self_area, np.logical_not(intersection))
        new_boundary = extract_boundary(new_area)
        self.area = csc_matrix(new_area)
        self.boundary = csc_matrix(new_boundary)
        self.outer_boundary = csc_matrix(new_boundary)
        self.hole = csc_matrix( np.logical_xor(self.area.toarray(), 
                                               self.boundary.toarray()) )
    
    def insert_hole(self, other):
        self_area = self.area.toarray()
        other_area = other.area.toarray()
        assert other_area in self_area
        new_area = np.logical_xor(self_area, other.hole.toarray())
        new_boundary = extract_boundary(new_area)
        self.area = csc_matrix(new_area)
        self.boundary = csc_matrix(new_boundary)
        self.hole = csc_matrix( np.logical_xor(self.area.toarray(), 
                                               self.boundary.toarray()) )
        self.subdomains.append( other )
        self.outer_boundary = self.extract_outer_boundary()
        
    def extract_outer_boundary(self):
        outer_boundary = self.boundary
        if len(self.subdomains) > 0:
            for n in range(len(self.subdomains)):
                outer_boundary -= self.subdomains[n].boundary
        return csc_matrix(outer_boundary)
            
class Rectangle(Shape):
    """A rectangle shape
    
    Args:
        X, Y: 2D arrays of coordinates produced from np.meshgrid(x,y) where x
              and y are 1D arrays of coordinates enclosing the full solution domain
              
        width, height: width and height of rectangle
        
        x0, y0: the center coordinates of the rectangle.
    """
    def __init__(self, 
                 X: np.ndarray, 
                 Y: np.ndarray, 
                 width: float, 
                 height: float,
                 x0: float=0.0, 
                 y0: float=0.0,
         ):
        super().__init__()
        self.center = (x0, y0)
        self.width = width
        self.height = height
        self.area = csc_matrix( (abs(X-x0) <= width/2) * (abs(Y-y0) <= height/2) )
        self.boundary = extract_boundary( self.area.toarray() )
        self.hole = csc_matrix( np.logical_xor(self.area.toarray(), 
                                               self.boundary.toarray()) )
        self.left_boundary = csc_matrix( self.boundary.toarray() * \
                                        (X == x0 - (width)/2) )
        self.right_boundary = csc_matrix( self.boundary.toarray() * \
                                          (X == x0 + (width)/2) )
        self.upper_boundary = csc_matrix( self.boundary.toarray() * \
                                        (Y == y0 + (height)/2) )
        self.lower_boundary = csc_matrix( self.boundary.toarray() * \
                                          (Y == y0 - (height)/2) )
        
        # corners
        lower_left_corner = (x0 - width/2, y0 - height/2)
        upper_left_corner = (x0 - width/2, y0 + height/2)
        upper_right_corner = (x0 + width/2, y0 + height/2)
        lower_right_corner = (x0 + width/2, y0 - height/2)
        self.corners = [lower_left_corner, upper_left_corner, 
                        upper_right_corner, lower_right_corner]
        self.corners_string = ["lower left: a", "upper left: b",
                               "upper right: c", "lower right: d"]
        
class Superellipse(Shape):
    """A superellipse shape defined by the following parametric equation:
        | (x-x0)/(w/2) |**p + | (y-y0)/(h/2) |**p <= 1. The power p defines the
        "roundness" of the corners, so if p = 2 you get an ellipse, p = 4 you get
        a traditional superellipse, if p --> infinity you get a rectangle.
        w and h denote the "total" width and height of the shape.
    
    Args:
        X, Y: 2D arrays of coordinates produced from np.meshgrid(x,y) where x
              and y are 1D arrays of coordinates enclosing the full solution domain
              
        width, height: width and height of shape
        
        dx, dy: the step-sizes in given directions
        
        x0, y0: the center coordinates of the shape.
    """
    def __init__(self, 
                 X: np.ndarray, 
                 Y: np.ndarray, 
                 width: float, 
                 height: float,
                 power: float,
                 dx: float,
                 dy: float,
                 x0: float, 
                 y0: float,
         ):
        super().__init__()
        self.center = (x0, y0)
        self.width = width
        self.height = height
        self.area = csc_matrix( abs((X-x0)/(width/2))**power + \
                    abs((Y-y0)/(height/2))**power <= 1 )
        self.boundary = extract_boundary( self.area.toarray() )
        self.hole = csc_matrix( np.logical_xor(self.area.toarray(), 
                                               self.boundary.toarray()) )

class Ellipse(Superellipse):
    """A regular ellipse shape. A subclass of Superellipse
    
    Args:
        X, Y: 2D arrays of coordinates produced from np.meshgrid(x,y) where x
              and y are 1D arrays of coordinates enclosing the full solution domain
              
        width, height: width and height of shape
        
        dx, dy: the step-sizes in given directions
        
        x0, y0: the center coordinates of the shape.
    """
    def __init__(self, 
                 X: np.ndarray, 
                 Y: np.ndarray, 
                 width: float, 
                 height: float,
                 dx: float,
                 dy: float,
                 x0: float, 
                 y0: float,
         ):
        super().__init__(X=X, Y=Y, width=width, height=height, power=2, dx=dx,
                         dy=dx, x0=x0, y0=x0)
        
class Circle(Ellipse):
    """A circle shape. A subclass of Ellipse
    
    Args:
        X, Y: 2D arrays of coordinates produced from np.meshgrid(x,y) where x
              and y are 1D arrays of coordinates enclosing the full solution domain
              
        width, height: width and height of shape
        
        dx, dy: the step-sizes in given directions
        
        x0, y0: the center coordinates of the shape.
    """
    def __init__(self, 
                 X: np.ndarray, 
                 Y: np.ndarray, 
                 diameter: float,
                 dx: float,
                 dy: float,
                 x0: float, 
                 y0: float,
         ):
        super().__init__(X=X, Y=Y, width=diameter, height=diameter, dx=dx, 
                         dy=dx, x0=x0, y0=x0)

###############################################################################
# Grid constructors
###############################################################################
# --- General-purpose non-uniform grid constructor --- #
def gridder(x_uniform: int=11,
            y_uniform: int=11,
            x_dense_points: list=[0], 
            y_dense_points: list=[0], 
            display_grid: bool=False,
            uniform: bool=True,
            x_res: list=[3,3], 
            y_res: list=[3,3]) -> list:
    """ 
    A general-purpose grid constructor. Supports both uniform and non-uniform
    grids. The non-uniform scheme uses a sequential arithmetic mean near
    the dense_points.
    
    Args:
        x_uniform: number of points in the x-direction (uniform grid)
        
        y_uniform: number of points in the y-direction (uniform grid)
        
        uniform: True or False depending on whether you want the grid to be
                 uniformly spaced or not. Setting to True requires specifying
                 x_dense_points, y_dense_points, etc.
        
        x_dense_points: a list of x-coordinates in between which to place
                        points. These are the "points of interest" for the
                        solution, for instance you may want to solve a
                        PDE for x in [0,1] but you want to make the grid
                        'denser' for 0.5<x<0.7, then you would use as input:
                            x_dense_points=[0, 0.5, 0.7, 1.0]
        
        y_dense_points: a list of y-coordinates in between which to place
                        points. These are the "points of interest" for the
                        solution, for instance you may want to solve a
                        PDE for y in [0,1] but you want to make the grid
                        'denser' for 0.5<y<0.7, then you would use as input:
                            y_dense_points=[0, 0.5, 0.7, 1.0]
        
        x_res: number of points in x-direction to place in between the 
               x_dense_points. For instance, setting x_res = [5] will place 5
               grid points in between each pair of x_dense_points inclusive
               of the x_dense_points (i.e. 3 points in between dense_points).
               You can also specify different number of points between 
               x_dense_points, for instance if x_dense_points=[0, 0.5, 0.7, 1.0],
               then setting x_res=[3,5,10] will put 3 points between 0<x<0.5,
               5 points between 0.5<x<0.7 and 10 points between 0.7<x<1.0, using
               uniformly spaced intervals (numpy linspace). All points in the 
               final grid will be inclusive of the x_dense_points. 
        
        y_res: number of points in x-direction to place in between the 
               y_dense_points. For instance, setting y_res = [5] will place 5
               grid points in between each pair of y_dense_points inclusive
               of the y_dense_points (i.e. 3 points in between dense_points).
               You can also specify different number of points between 
               y_dense_points, for instance if y_dense_points=[0, 0.5, 0.7, 1.0],
               then setting y_res=[3,5,10] will put 3 points between 0<y<0.5,
               5 points between 0.5<y<0.7 and 10 points between 0.7<y<1.0, using
               uniformly spaced intervals (numpy linspace). All points in the 
               final grid will be inclusive of the y_dense_points.
    
    Outputs:
        X, Y: the meshgrid grid matrices
        
        x, y: vectors of x and y coordinates for the grid
        
        dx, dy: vectors of spacings (step-sizes) in x and y-directions. Note 
                that if x has Nx elements, dx has Nx-1 elements, similarly if
                y has Ny elements, dy has Ny-1 elements.
    """
    assert type(x_res) == list
    assert type(y_res) == list
    
    # --- x vector --- #
    dense_points = x_dense_points
    N_uniform = x_uniform
    N = len(dense_points)
    xlist = list(np.linspace(dense_points[0], dense_points[-1], N_uniform))
        
    if not uniform:
        xlist = []
        if len(x_res) == 1:
            x_res = [x_res[0]]*(len(dense_points) - 1)
        elif len(x_res) == 2:
            x0, x1 = x_res[0], x_res[1]
            x_res = []
            for ii in range(len(dense_points)-1):
                if ii % 2 == 0: 
                    x_res.append(x0)
                else:
                    x_res.append(x1)
        elif len(x_res) > 2:
            assert len(x_res) == len(dense_points)-1
            
        if len(dense_points) > 1:
            for jj in range(N-1):
                xL = dense_points[jj]
                xR = dense_points[jj+1]
                xM = list(np.linspace(xL,xR,x_res[jj]+2)[1:-1])
                if xL not in xlist: xlist.append(xL)
                xlist = xlist + xM
                if xR not in xlist: xlist.append(xR)
            x_new = np.sort(np.array(xlist))
            xlist = list(x_new)
    
    # finish array
    x = np.sort(np.array(xlist))
    dx = np.diff(x) # array of spacings
    
    # --- y vector --- #
    dense_points = y_dense_points
    N_uniform = y_uniform
    N = len(dense_points)
    ylist = list(np.linspace(dense_points[0], dense_points[-1], N_uniform))
    
    if not uniform:
        ylist = []
        if len(y_res) == 1:
            y_res = [y_res[0]]*(len(dense_points) - 1)
        elif len(y_res) == 2:
            y0, y1 = y_res[0], y_res[1]
            y_res = []
            for ii in range(len(dense_points)-1):
                if ii % 2 == 0: 
                    y_res.append(y0)
                else:
                    y_res.append(y1)
        elif len(y_res) > 2:
            assert len(y_res) == len(dense_points)-1
            
        assert len(y_res) == len(dense_points)-1
        if len(dense_points) > 1:
            for jj in range(N-1):
                yL = dense_points[jj]
                yR = dense_points[jj+1]
                yM = list(np.linspace(yL,yR,y_res[jj]+2)[1:-1])
                if yL not in ylist: ylist.append(yL)
                ylist = ylist + yM
                if yR not in ylist: ylist.append(yR)
            y_new = np.sort(np.array(ylist))
            ylist = list(y_new)
    
    # finish array
    y = np.sort(np.array(ylist))
    dy = np.diff(y) # array of spacings
    
    # Combine
    [X,Y] = np.meshgrid(x, y)
    
    if display_grid:
        fig = plt.figure(dpi=600)
        plt.scatter(X,Y,s=2.5,c="black")
        plt.xlim([min(x),max(x)])
        plt.ylim([min(y),max(y)])
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.title("Grid points")
        plt.tight_layout()
        fig.gca().set_aspect('equal')
    
    return [X, Y, x, y, dx, dy]

###############################################################################
# Derivatives
###############################################################################
"""
Based on the paper:
    Sundqvist, H. and Veronis, G., 1970. A simple finite‐difference grid 
    with non‐constant intervals. Tellus, 22(1), pp.26-31.
"""
# --- 1st derivatives --- #
def Dx(x: np.ndarray, y: np.ndarray, dx: np.ndarray):
    assert(len(dx) == len(x)-1)
    Ny = len(y)
    nb_Dx() # compile
    B = nb_Dx(x=x, dx=dx)
    D = sp.sparse.block_diag( [B]*Ny, format="csc" )
    return D

@nb.njit(parallel=True)
def nb_Dx(x: np.ndarray=np.array([-1,0,1]), 
          dx: np.ndarray=np.array([1,1])):
    Nx = len(x)
    Dblock = np.zeros( (Nx,Nx) )
    
    # 1st row
    h_minus = dx[0]
    h_plus = dx[0]
    Dblock[0,1] = 1/(h_plus + h_minus)
    
    # last row
    h_minus = dx[-1]
    h_plus = dx[-1]
    Dblock[-1,-2] = -1/(h_plus + h_minus)
    
    # in-between rows
    for ii in nb.prange(1,Nx-1):
        h_minus = dx[ii-1]
        h_plus = dx[ii]
        Dblock[ii,ii-1] = -1/(h_plus + h_minus)
        Dblock[ii,ii+1] = 1/(h_plus + h_minus)
    
    return Dblock

def Dy(x: np.ndarray, y: np.ndarray, dy: np.ndarray):
    assert(len(dy) == len(y)-1)
    Nx = len(x)
    nb_Dy() # compile
    [BL, BR] = nb_Dy(x=x, y=y, dy=dy)
    D = sp.sparse.diags([BR, BL], [+Nx,-Nx], format="csc")
    return D

@nb.njit(parallel=True)
def nb_Dy(x: np.ndarray=np.array([-1,0,1]),
          y: np.ndarray=np.array([-1,0,1]), 
          dy: np.ndarray=np.array([1,1])):
    Nx = len(x)
    Ny = len(y)
    off_diag_right = np.zeros( (Nx*Ny-Nx,) )
    off_diag_left = np.zeros( (Nx*Ny-Nx,) )
    
    # 1st block
    h_minus = dy[0]
    h_plus = dy[0]
    off_diag_right[:Nx] = 1/(h_plus + h_minus)
    
    # in-between rows
    for ii in nb.prange(1,Ny-1):
        h_minus = dy[ii-1]
        h_plus = dy[ii]
        off_diag_left[(ii-1)*Nx:ii*Nx] = -1/(h_plus + h_minus)
        off_diag_right[ii*Nx:(ii+1)*Nx] = 1/(h_plus + h_minus)
    
    # last block
    h_minus = dy[-1]
    h_plus = dy[-1]
    off_diag_left[(Ny-1)*Nx-Nx:] = -1/(h_plus + h_minus)
    
    return [off_diag_left, off_diag_right]

# --- 2nd derivatives --- #
def Dxx(x: np.ndarray, y: np.ndarray, dx: np.ndarray):
    assert(len(dx) == len(x)-1)
    Ny = len(y)
    nb_Dxx() # compile
    B = nb_Dxx(x=x, dx=dx)
    D = sp.sparse.block_diag( [B]*Ny, format="csc" )
    return D

@nb.njit(parallel=True)
def nb_Dxx(x: np.ndarray=np.array([-1,0,1]), 
           dx: np.ndarray=np.array([1,1])):
    Nx = len(x)
    Dblock = np.zeros( (Nx,Nx) )
    
    # 1st row
    h_minus = dx[0]
    h_plus = dx[0]
    Dblock[0,0] = -2/(h_plus*h_minus)
    Dblock[0,1] = 2/h_plus/(h_plus + h_minus)
    
    # last row
    h_minus = dx[-1]
    h_plus = dx[-1]
    Dblock[-1,-2] = 2/h_minus/(h_plus + h_minus)
    Dblock[-1,-1] = -2/(h_plus*h_minus)
    
    # in-between rows
    for ii in nb.prange(1,Nx-1):
        h_minus = dx[ii-1]
        h_plus = dx[ii]
        Dblock[ii,ii-1] = 2/h_minus/(h_plus + h_minus)
        Dblock[ii,ii] = -2/(h_plus*h_minus)
        Dblock[ii,ii+1] = 2/h_plus/(h_plus + h_minus)
    
    return Dblock

def Dyy(x: np.ndarray, y: np.ndarray, dy: np.ndarray):
    assert(len(dy) == len(y)-1)
    Nx = len(x)
    nb_Dyy() # compile
    [BL, BM, BR] = nb_Dyy(x=x, y=y, dy=dy)
    D = sp.sparse.diags([BR, BM, BL], [+Nx,0,-Nx], format="csc")
    return D

@nb.njit(parallel=True)
def nb_Dyy(x: np.ndarray=np.array([-1,0,1]),
           y: np.ndarray=np.array([-1,0,1]), 
           dy: np.ndarray=np.array([1,1])):
    Nx = len(x)
    Ny = len(y)
    main_diag = np.zeros( (Nx*Ny,) )
    off_diag_right = np.zeros( (Nx*Ny-Nx,) )
    off_diag_left = np.zeros( (Nx*Ny-Nx,) )
    
    # 1st block
    h_minus = dy[0]
    h_plus = dy[0]
    main_diag[:Nx] = -2/(h_plus*h_minus)
    off_diag_right[:Nx] = 2/h_plus/(h_plus + h_minus)
    
    # in-between rows
    for ii in nb.prange(1,Ny-1):
        h_minus = dy[ii-1]
        h_plus = dy[ii]
        off_diag_left[(ii-1)*Nx:ii*Nx] = 2/h_minus/(h_plus + h_minus)
        main_diag[ii*Nx:(ii+1)*Nx] = -2/(h_plus*h_minus)
        off_diag_right[ii*Nx:(ii+1)*Nx] = 2/h_plus/(h_plus + h_minus)
    
    # last block
    h_minus = dy[-1]
    h_plus = dy[-1]
    off_diag_left[(Ny-1)*Nx-Nx:] = 2/h_minus/(h_plus + h_minus)
    main_diag[(Ny-1)*Nx:] = -2/(h_plus*h_minus)
    
    return [off_diag_left, main_diag, off_diag_right]

def Laplacian(x: np.ndarray, y: np.ndarray, dx: np.ndarray, dy: np.ndarray):
    return Dxx(x=x, y=y, dx=dx) + Dyy(x=x, y=y, dy=dy)

def Grad(x: np.ndarray, y: np.ndarray, dx: np.ndarray, dy: np.ndarray,
         f: np.ndarray):
    if type(f) is not np.ndarray: f = f.toarray()
    xv = (Dx(x=x, y=y, dx=dx) @ f.flatten()[:,None]).reshape(np.shape(f))
    yv = (Dy(x=x, y=y, dy=dy) @ f.flatten()[:,None]).reshape(np.shape(f))
    return [xv, yv]

###############################################################################
# Boundary points extraction functions
###############################################################################
def extract_boundary(A: np.ndarray) -> np.ndarray:
    """Given a binary matrix A containing some filled shape(s) (e.g. circle), this
    function determines which points (pixels) make up the boundary enclosing said
    shape(s), and outputs a binary matrix containing those boundary points only.
    
    Note: it is important that A is a completely filled shape, otherwise the
    detected boundary will include inner points (e.g. the boundary of subdomains)
    and this may not be desirable for the user.
    
    Args:
        A: the binary matrix describing the shape/geometry of the domain
    """
    A_trial = np.ones((3,3)); find_edges(A_trial) # compilation step

    # Extract points from outer frame 
    if type(A) is np.matrix: A = np.array(A)
    edge = np.zeros(np.shape(A))
    edge[0,:] = 1
    edge[-1,:] = 1
    edge[:,0] = 1
    edge[:,-1] = 1
    outer_frame = edge.astype(bool) * A
        
    # Define boundary
    boundary = find_edges(A).astype(bool) | outer_frame 
    
    return csc_matrix(boundary)

@nb.njit(parallel=True)
def find_edges(A: np.ndarray) -> np.ndarray:
    (Ny, Nx) = np.shape(A)
    B = np.zeros(np.shape(A))
    for ii in nb.prange(1,Ny-1):
        for jj in nb.prange(1,Nx-1):
            condition = A[ii-1,jj] and A[ii+1,jj] and A[ii,jj+1] and \
                        A[ii,jj-1] and A[ii+1,jj+1] and A[ii-1,jj+1] and\
                        A[ii+1,jj-1] and A[ii-1,jj-1]
                        
            if A[ii,jj] and not condition: B[ii,jj] = 1
    return B

###############################################################################
# Linear PDE solver
###############################################################################
def solve_pde(X: np.ndarray,
              Y: np.ndarray,
              dx: float,
              dy: float,
              domain: np.ndarray,
              inner_domain: np.ndarray,
              boundary: list,
              boundary_values: list,
              LHS: np.ndarray,
              source: np.ndarray = None,
              plot_solution: bool=True) -> np.ndarray:
    """Solves a boundary-value problem of the form LHS*u = source
    where LHS is the left-hand side operator, source is the source function 
    (optional) and return u(x,y) as the solution over the prescribed domain,
    and given the boundary values.
    
    Args:
        X, Y: meshgrid matrices of coordinates for a rectangle of size Nx by Ny,
              where Nx is the number of points in x-direction and Ny is the number
              of points in the y-direction
              
        dx, dy: step-size in each direction
        
        domain: a (Ny,Nx) binary matrix in csc_matrix sparse format (scipy sparse)
        
        inner_domain: a (Ny,Nx) binary matrix similar to domain but excluding boundary 
                      points
                      
        boundary: (Ny,Nx) binary matrix containing boundary point locations in sparse
                  csc_matrix format
                  
        boundary_values: the boundary condition values in sparse csc_matrix format of
                         dimension (Ny,Nx)
                         
        LHS: left-hand side operator matrix in sparse csc_matrix format of dimension
             (Ny,Nx)
             
        source: source function in sparse csc_matrix format of dimension (Ny,Nx)
        
        plot_solution: True if you want plots
    """
    # Start timer
    start_time = time.time()
    
    # Handle source
    if source is None: source = np.zeros(np.shape(X))
    
    # Apply boundary conditions
    N = len(boundary)
    u_boundary = 0.0
    boundary_points = 0
    for n in range(N):
        u_boundary += csc_matrix( boundary[n].multiply(csc_matrix(boundary_values[n])) )
        boundary_points += boundary[n]
    
    # Solve system
    if source is not sp.sparse._csc.csc_matrix: source = csc_matrix(source)
    solution = linear_solver(LHS = LHS, 
                             source = source, 
                             inner_domain = inner_domain,
                             boundary = boundary_points, 
                             boundary_values = u_boundary)
    
    # Close timer
    final_time = time.time() - start_time
    print("#----- computation time = %s seconds -----#" %(np.round(final_time,2)))
    
    # plot solutions (optional)
    if plot_solution:
        CM = 'RdBu'
        
        fig, ax = plt.subplots( figsize=(6,6), dpi=500 )
        c = ax.pcolor(X, Y, u_boundary.toarray(), cmap=CM)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_title(r"$u|_{\partial\Omega}(x,y)$")
        ax.grid(visible=None)
        fig.colorbar(c, ax=ax)
        fig.tight_layout()
        plt.show()
        
        fig, ax = plt.subplots( figsize=(6,6), dpi=500 )
        c = ax.pcolor(X, Y, solution.toarray(), cmap=CM)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_title(r"$u(x,y)$")
        ax.grid(visible=None)
        fig.colorbar(c, ax=ax)
        fig.tight_layout()
        plt.show()
    
    return solution

def linear_solver(LHS: np.ndarray, 
                  source: np.ndarray, 
                  inner_domain: np.ndarray,
                  boundary: np.ndarray,
                  boundary_values: np.ndarray):
    """Given a system of equations of the form L*u = b where L is the LHS operator,
    u is the solution vector and b is the source vector, we insert the boundary values
    for u by re-arranging the linear system and removing redundancies. 
    
    Suppose that we start with N equations and LHS has size [N x N], and then
    we specify m boundary values at boundary where m < N, it follows that
    there are now N-m unknowns u, and so we want to create a reduced system of
    N-M equations. 
    
    The outputs of this function are a reduced LHS matrix of size [(N-m) x (N-m)],
    and a RHS column vector of size [(N-m) x 1], where RHS contains both the
    source values and the boundary conditions applied at the points OTHER than
    at the boundary points.
    
    If we let the reduced system be defined as L'*u' = b', then 
    u' = inverse(L')*b' will give us the values of u at all the points OTHER
    than at the boundary points. This means that to recover the full solution
    for u we must then parse u' with all the boundary values we started with,
    which can be done with a separate function called reconstruct_system().
    """
    assert type(LHS) is sp.sparse._csc.csc_matrix
    assert type(boundary) is sp.sparse._csc.csc_matrix
    assert type(boundary_values) is sp.sparse._csc.csc_matrix
    assert type(source) is sp.sparse._csc.csc_matrix
    N = np.prod( boundary.shape )
    
    boundary_vector = boundary.toarray().flatten()[None,:]
    BC_vector = boundary_values.toarray().flatten()[:,None]
    inner_domain_vector = inner_domain.toarray().flatten()[None,:]
    source_in = source.toarray().flatten()[:,None]
    
    # Move boundary values to RHS of equation system  
    LHS_mask = csc_matrix( LHS.multiply(boundary_vector) )
    RHS = csc_matrix( source_in - LHS_mask @ BC_vector )
    
    # remove columns
    LHS_reduced_cols = \
        csc_matrix( LHS.transpose()[inner_domain_vector[0,:]].transpose() )
    
    # remove rows
    LHS_system = LHS_reduced_cols[inner_domain_vector[0,:]]
    RHS_system = RHS[inner_domain_vector[0,:]]
    print("# of equations to solve: %s out of %s grid points" %(LHS_system.shape[0],
                                                                N))

    # Solve u(x,y) inside domain
    solution_inner = sp.sparse.linalg.spsolve(LHS_system, RHS_system)
    print("Solution on inner points complete. Reconstructing full system...")

    # Full solution
    full_solution = np.zeros((N,))
    full_solution[inner_domain_vector[0,:]] = solution_inner
    full_solution[boundary_vector[0,:]] = BC_vector[:,0][boundary_vector[0,:]]
    
    full_solution = np.reshape( full_solution, np.shape(boundary) )
    print("Done.")
    
    return csc_matrix( full_solution )

###############################################################################
# Other useful functions
###############################################################################    
def plot_matrix(A):
    if type(A) is sp.sparse._csc.csc_matrix: 
        A_input = A.toarray()
    else:
        A_input = A
    plt.matshow(A_input); plt.colorbar()
    return 0

def plot_stream(X:np.ndarray, Y: np.ndarray, f: np.ndarray, stream: np.ndarray, 
                color: str="white"):
    CM = 'RdBu'
    if type(stream[0]) is not np.ndarray: stream[0] = stream[0].toarray()
    if type(stream[1]) is not np.ndarray: stream[1] = stream[1].toarray()
    if type(f) is not np.ndarray: f = f.toarray()

    fig, ax = plt.subplots( figsize=(6,6), dpi=500 )
    c = ax.pcolor(X, Y, f, cmap=CM)
    ax.quiver(X, Y, stream[0], stream[1], color=color)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(r"$\nabla u(x,y)$")
    ax.grid(visible=None)
    fig.colorbar(c, ax=ax)
    fig.tight_layout()
    plt.show()

###############################################################################
# Test functions
###############################################################################
if __name__ == "__main__":
    #%% Example grid
    r = 1.0
    [X,Y,x,y,dx,dy] = gridder(x_uniform = 11, 
                              y_uniform = 11,
                              display_grid = False,
                              uniform = False,
                              x_dense_points = [-2*r,-r,r,2*r], 
                              y_dense_points = [-2*r,-r,r,2*r], 
                              x_res = [3,20], 
                              y_res = [3,20])
    
    Rec = Rectangle(X=X, Y=Y, width=2*r, height=2*r, x0=0.0, y0=0.0)
    boundary = Rec.boundary

    fig = plt.figure(dpi=600)
    plt.scatter(X, Y, s=3, color="black")
    plt.scatter(X[Rec.boundary.toarray()], Y[Rec.boundary.toarray()],
                s=3, color="red")
    plt.xlim([min(x), max(x)])
    plt.ylim([min(y), max(y)])
    plt.tight_layout()
    plt.gca().set_aspect('equal')