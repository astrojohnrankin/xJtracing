import numpy as np

# TEMPORARY
import os, sys
# sys.path.append(os.path.dirname(__file__))
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append("..")

from xJtracing.tracing_utils import assert_single_number
from xJtracing.Lobster_ray_tracing import Lobster_derived_geometrical_parameters

def generate_tubes(configuration, cell_centers, radius, complete_vectorization):
    """
    Generates the capillaries (for Lobster Angel), or logical equivalent in case of Lobster Schimdt.

    Notes
    -----
    See define_Lobster_mirrors's documentation for the equations derivation.
    """
    if configuration in ['Angel']: assert complete_vectorization
    
    if complete_vectorization:
        x0_tube, y0_tube = np.meshgrid(cell_centers, cell_centers)
        x0_tube, y0_tube = x0_tube.flatten(), y0_tube.flatten()
    else:
        if configuration in ['Schmidt_top']:
            x0_tube = np.repeat(0, cell_centers.size)
            y0_tube = cell_centers
        if configuration in ['Schmidt_bottom']:
            x0_tube = cell_centers
            y0_tube = np.repeat(0, cell_centers.size)
        
    alpha = np.arctan(np.sqrt(x0_tube**2 + y0_tube**2)/radius)
    theta = np.arctan2(y0_tube, x0_tube)    
    
    return x0_tube, y0_tube, alpha, theta


def generate_cells_arranged_equal(configuration, cell_distance, cell_side, cell_length, number_cells_per_side, radius_curvature, complete_vectorization=True):
    """
    Function to generate the cells of a Lobster with all cells equal.

    Parameters
    ----------
    configuration: str
        Configuration of the Lobster.
    cell_distance: float
        Distance between the center of two cells.
    cell_side: float
        Lenght of the side of each cell.
    cell_length: float
        Mirror length of each cell.
    number_cells_per_side: int
        How many cells are on each side (so that for an Angel Lobster the number of cells would be number_cells_per_side**2, 
        but not for Schmidt).
    radius_curvature: float
        Curvature radius of the KB layer.
    complete_vectorization: bool
        Experimental, if True, the numpy calculations are vectorized (but in this case each 'cell' can only hit the one in front).
    """
    cell_centers = np.arange(-cell_distance*number_cells_per_side/2, cell_distance*number_cells_per_side/2+cell_distance, cell_distance)
    x0_tube, y0_tube, alpha, theta = generate_tubes(configuration, cell_centers, radius_curvature, complete_vectorization=complete_vectorization)
    sides = np.tile(cell_side, x0_tube.size)
    L_low  = np.tile(0, x0_tube.shape)
    L_high = np.tile(cell_length, x0_tube.shape)

    return alpha, theta, sides, L_low, L_high


def generate_cells_arranged_variable_L(configuration, cell_distance, cell_side, cell_length, number_cells_per_side, radius_curvature, complete_vectorization=True):
    """
    Function to generate the cells of a Lobster with a variable mirror length to avoid on-axis double reflection on opposite sides.

    Parameters
    ----------
    configuration: str
        Configuration of the Lobster.
    cell_distance: float
        Distance between the center of two cells.
    cell_side: float
        Lenght of the side of each cell.
    cell_length: float
        Mirror length of each cell.
    number_cells_per_side: int
        How many cells are on each side (so that for an Angel Lobster the number of cells would be number_cells_per_side**2, 
        but not for Schmidt).
    radius_curvature: float
        Curvature radius of the Lobster telescope or telescope layer.
    complete_vectorization: bool
        Experimental, if True, the numpy calculations are vectorized (but in this case each 'cell' can only hit the one in front).
    """
    cell_centers = np.arange(-cell_distance*number_cells_per_side/2, cell_distance*number_cells_per_side/2+cell_distance, cell_distance)
    x0_tube, y0_tube, alpha, theta = generate_tubes(configuration, cell_centers, radius_curvature, complete_vectorization=complete_vectorization)
    sides = np.tile(cell_side, x0_tube.size)
    L_low = cell_length - sides/np.tan(alpha)
    L_low[alpha == 0] = 0
    L_low[L_low < 0] = 0  
    L_high = np.tile(cell_length, x0_tube.shape)

    return alpha, theta, sides, L_low, L_high

   
def define_Lobster_mirrors(configuration, reflecting, radius,
                                            cell_side, cell_length, cell_distance, 
                                            number_cells_per_side, radius_inner=None, max_reflections=2, x0=0, y0=0, 
                           cells_arrengement_function=generate_cells_arranged_equal, complete_vectorization=True, angle_min=None, angle_max=None):
    """
    Defines the geometrical parameters of an Angel Lobster telescope, or of one layer of a Schmidt Lobster telescope.

    Parameters
    ----------
    configuration: str
        Configuration of the telescope or telescope layer, can be 'Angel','Schmidt_top', 'Schmidt_bottom'
    reflecting: str
        If total all sides of mirrors are reflecting, while if partial only inner sides of mirrors are reflecting.
    radius: float
        Curvature radius of the Lobster telescope or telescope layer.
    cell_side: float
        Lenght of the side of each cell.
    cell_length: float
        Mirror length of each cell.
    cell_distance: float
        Distance between the center of two cells.
    number_cells_per_side: int
        How many cells are on each side (so that for an Angel Lobster the number of cells would be number_cells_per_side**2, 
        but not for Schmidt).
    radius_inner: float
        Radius of the inner mirror in case of a top Schmidt telescope layer.
    max_reflections: int
        Maximum number of reflections simulated.
    x0, y0: float
        Origin of the telescope reference system.
    cells_arrengement_function: function
        Function generating the parameters of the cells.
    complete_vectorization: bool
        Experimental, if True, the numpy calculations are vectorized (but in this case each 'cell' can only hit the one in front).
    angle_min, angle_max: float
        Angles in which the mirrors are generated.


    Notes
    -----
    To build a Lobster eye we start with a flat surface, described as
    
    .. math::  
		\\begin{cases}
		x=\\tan\\left(\\alpha\\right)\\left(z-z_{0}\\right)\\cos\\left(\\theta\\right)+\\sin\\left(\\theta\\right)t+x_{0}\\\\
		y=\\tan\\left(\\alpha\\right)\\left(z-z_{0}\\right)\\sin\\left(\\theta\\right)+\\cos\\left(\\theta\\right)t+y_{0}
		\\end{cases}
    
    where :math:`\\alpha` and :math:`\\theta` are respectively the inclination and
    azimuthal angle of the flat surface, and :math:`t` is a parametric variable
    used to plot the surface; the surface is so designed to pass through
    the :math:`\\left(x_{0},y_{0},z_{0}\\right)` point.
    
    Given a grid of points (the centers of each cell), considering the
    spherical surface we have
    
    .. math::  
		\\alpha=\\arctan\\left(\\frac{\\sqrt{x_{0\\;tube}^{2}+y_{0\\;tube}^{2}}}{R}\\right)
    
    .. math::  
		\\theta=\\arctan\\left(\\frac{x_{0\\;tube}}{y_{0\\;tube}}\\right)
    
    where :math:`x_{0\\;tube}` and :math:`y_{0\\;tube}` are the coordinates of points,
    at vertical coordinate :math:`z=R`, through which lines perpendicular to
    the spherical surface of the Lobster pass, and :math:`R` is the curvature
    radius of the Lobster surface. The inclinations over the two axis
    are then 
    
    .. math::  
		\\begin{cases}
		\\alpha_{x}=\\arctan\\left(\\tan\\left(\\alpha\\right)\\cos\\left(\\theta\\right)\\right)\\\\
		\\alpha_{y}=\\arctan\\left(\\tan\\left(\\alpha\\right)\\sin\\left(\\theta\\right)\\right)
		\\end{cases}
    
    the sides of each cell are then given by 
    
    .. math::  
		\\begin{cases}
		S_{x}=\\frac{S}{\\cos\\left(\\alpha_{x}\\right)}\\\\
		S_{y}=\\frac{S}{\\cos\\left(\\alpha_{y}\\right)}
		\\end{cases}
    
    in the Schmidt Lobster, of course, the cells are not square but one
    side in each layer is long as the entire telescope.
    
    The walls are positioned so that their planes pass through the coordinates
    :math:`x_{0}\\pm\\frac{S_{x}}{2}` and :math:`y_{0}\\pm\\frac{S_{y}}{2}` depending
    on their orientation. The walls are then defined by defining their
    borders in three dimensions, the vertical :math:`\\left(z\\right)` borders
    are defined as 
    
    .. math::  
		R_{curvature}\\cos\\left(\\alpha\\right)-z_{offset}<z<\\left(R_{curvature}+L\\right)\\cos\\left(\\alpha\\right)-z_{offset}
    
    where :math:`R` is the radial distance from the center, :math:`L` is the length
    of each tube, and 
    
    .. math::  
		z_{offset}=R_{curvature}-R
    
    is an offset: for an Angel Lobster and for the bottom Schmidt array,
    we have that :math:`R_{curvature}=R`, but for the top Schmidt array, the
    radial distance from the center reference is not the same as the curvature
    radius of the spherical Lobster surface, in this case
    
    .. math::  
		R_{curvature}=2\\left(R-\\frac{R_{inner}}{2}\\right)
    
    where :math:`R_{inner}` is the radius of the lower layer (basically this
    way we compute 2 times the distance to the focus of the inner radius).
    
    In the :math:`x` and :math:`y` plane the boundaries considered are
    
    .. math::  
		x_{0}-\\frac{S_{x}}{2}+z\\tan\\left(\\alpha_{x}\\right)<x(z)<x_{0}+\\frac{S_{x}}{2}+z\\tan\\left(\\alpha_{x}\\right)
 
		y_{0}-\\frac{S_{y}}{2}+z\\tan\\left(\\alpha_{y}\\right)<y(z)<y_{0}+\\frac{S_{y}}{2}+z\\tan\\left(\\alpha_{y}\\right)
    
    """
    assert reflecting in ['total', 'partial'], print(reflecting)
    assert configuration in ['Angel','Schmidt_top', 'Schmidt_bottom']

    dict_tot = {}
    
    nord_sud_configurations = ['Angel', 'Schmidt_top']
    est_ovest_configurations = ['Angel', 'Schmidt_bottom']
    assert_single_number(radius, cell_side, cell_length, cell_distance, number_cells_per_side, x0, y0)
    
    if configuration == 'Schmidt_top': 
        radius_curvature = 2*(radius - radius_inner/2) #poi ci dovrò scrivere la dimostrazione (non come commento ma nella documentazione)
    else:
        assert radius_inner == None
        radius_curvature = radius


    alpha, theta, sides, L_low, L_high = cells_arrengement_function(configuration, cell_distance, cell_side, cell_length, number_cells_per_side, radius_curvature, complete_vectorization)
    
    alpha_x, alpha_y, x_at_top, y_at_top = Lobster_derived_geometrical_parameters(alpha, theta, x0, y0, L_high, radius)

    sides_x = sides/np.cos(alpha_x)
    sides_y = sides/np.cos(alpha_y)
    dict_tot['side_long'] = number_cells_per_side*cell_distance
    if not complete_vectorization:
        if configuration in ['Schmidt_top']:
            sides_x = np.repeat(number_cells_per_side*cell_distance, alpha_y.size)
        elif configuration in ['Schmidt_bottom']:
            sides_y = np.repeat(number_cells_per_side*cell_distance, alpha_x.size)

    if angle_min or angle_max:
        sides_x[np.logical_or(np.abs(alpha)<angle_min, np.abs(alpha)>angle_max)] = 0
        sides_y[np.logical_or(np.abs(alpha)<angle_min, np.abs(alpha)>angle_max)] = 0

    if configuration in ['Angel']:
        dict_tot['cardinal_keys'] = ['nord', 'sud', 'est', 'ovest']
    elif configuration in ['Schmidt_top']:
        dict_tot['cardinal_keys'] = ['nord', 'sud']
    elif configuration in ['Schmidt_bottom']:
        dict_tot['cardinal_keys'] = ['est', 'ovest']

    if reflecting=='partial':
        def determine_nonreflecting_sides(surface_key):
            surface_exists = np.repeat(True, sides_x.shape)
            if configuration in nord_sud_configurations:
                if surface_key == 'nord':
                    surface_exists[y_at_top < 0] = False
                elif surface_key == 'sud':
                    surface_exists[y_at_top > 0] = False
            if configuration in est_ovest_configurations:
                if surface_key == 'est':
                    surface_exists[x_at_top < 0] = False
                elif surface_key == 'ovest':
                    surface_exists[x_at_top > 0] = False
            return surface_exists
        
        dict_tot['surface_is_reflecting'] = {}
        for cardinal_key in dict_tot['cardinal_keys']:
            dict_tot['surface_is_reflecting'][cardinal_key] = determine_nonreflecting_sides(cardinal_key)
    else:
        dict_tot['surface_is_reflecting'] = 'to be computed later'


    dict_tot['alpha'] = alpha
    dict_tot['theta'] = theta
    dict_tot['sides_x'] = sides_x
    dict_tot['sides_y'] = sides_y
    dict_tot['radius_curvature'] = radius_curvature
    dict_tot['radius'] = radius
    dict_tot['L_low'] = L_low
    dict_tot['L_high'] = L_high
    dict_tot['x0'] = x0
    dict_tot['y0'] = y0
    dict_tot['reflecting'] = reflecting
    dict_tot['max_reflections'] = max_reflections
    dict_tot['n_configs'] = alpha.size
    return dict_tot
    