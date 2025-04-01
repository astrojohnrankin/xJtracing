import numpy as np
import matplotlib.pyplot as plt

import sys
import os
# sys.path.append(os.path.dirname(__file__))
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append("..")


from xJtracing.mirror import generate_KB_parabola, KB_parabola_dataclass


def cell_centers_sides_equal_adaptiveL(cell_distance, number_cells_per_side, cell_side):
    """
    Find the centers of cells, and their sides, in case of a KB with equal or adaptive mirror length.
    """
    cell_centers = np.arange(-cell_distance*number_cells_per_side/2, cell_distance*number_cells_per_side/2+cell_distance, cell_distance)
    sides = np.tile(cell_side, cell_centers.size)
    return cell_centers, sides

def cell_centers_sides_adaptiveside(cell_distance, number_cells_per_side, cell_side, f, radius, cell_length):
    """
    Find the centers of cells, and their sides, in case of a KB with adaptive mirror side.
    """
    fine = 0
    cell_centers = [0]
    sides = []
    nuovo_side = cell_side
    while fine < number_cells_per_side*cell_distance/2:

        def mirror_surface_x(f, L, R0):
            def mirror_surface_x_(z):
                f0 = (np.sqrt((f-L)**2 + R0**2/2) - (L - f))/2
                return np.sqrt(8*f0*(z - f + f0))
            return mirror_surface_x_
        spessore = cell_distance - cell_side

        parabola_riflettente = mirror_surface_x(f, radius, fine)
        B = parabola_riflettente(radius + cell_length)
        nuovo_side = B - fine
        if nuovo_side < cell_side: nuovo_side = cell_side
        nuovo_centro = fine + nuovo_side/2
        fine += nuovo_side + spessore

        cell_centers.append(nuovo_centro)
        sides.append(nuovo_side)

    cell_centers = cell_centers[1:]
    cell_centers = np.array([*np.flip(-np.array(cell_centers)), *cell_centers])
    sides = np.array([*np.flip(np.array(sides)), *sides])
    return cell_centers, sides


def L_bottom_adaptiveL(f, L_top, cell_centers, sides, radius):
    """
    Bottom height of a KB mirror with adaptive length.
    """
    _, _, parabola_plus = generate_KB_parabola(f=f, L=L_top, R0=cell_centers + sides/2, axis='x')
    _, _, parabola_minus = generate_KB_parabola(f=f, L=L_top, R0=cell_centers - sides/2, axis='x') 

    L_plus = parabola_plus(cell_centers - sides/2, None)
    L_minus =  parabola_minus(cell_centers + sides/2, None)
    
    L_bottom = np.empty(cell_centers.size)
    L_bottom[cell_centers < 0] = L_minus[cell_centers < 0]
    L_bottom[cell_centers > 0] = L_plus[cell_centers > 0]
    L_bottom[L_bottom < radius] = radius
    L_bottom[int((cell_centers.size-1)/2)] = radius
    return L_bottom

def L_top_compute(radius, cell_length, cell_centers):
    """
    Top height of a KB mirror.
    """
    return np.repeat(radius + cell_length, cell_centers.size)


def generate_KB_cells_equal(cell_distance, number_cells_per_side, cell_side, cell_length, radius, f):
    """
    Function to generate the cells of a KB with all cells equal. Note, we call each aperture of the KB cell, in analogy with 
    an Angel Lobster, even if this design is different.

    Parameters
    ----------
    cell_distance: float
        Distance between the center of two cells.
    number_cells_per_side: int
        How many cells are on each side (so that for an Angel Lobster the number of cells would be number_cells_per_side**2, 
        but not in this case)
    cell_side: float
        Lenght of the side of each cell.
    cell_length: float
        Mirror length of each cell.
    radius: float
        Curvature radius of the KB layer.
    f: float
        Focal length of the KB layer.
    """
    cell_centers, sides = cell_centers_sides_equal_adaptiveL(cell_distance, number_cells_per_side, cell_side)
    L_top = L_top_compute(radius, cell_length, cell_centers)
    L_bottom = np.repeat(radius, cell_centers.size)
    return cell_centers, sides, L_bottom, L_top

def generate_KB_cells_adaptive_L(cell_distance, number_cells_per_side, cell_side, cell_length, radius, f):
    """
    Function to generate the cells of a KB with adaptive mirror length so to have a complete filling factor. 
    Note, we call each aperture of the KB cell, in analogy with an Angel Lobster, even if this design is different.

    Parameters
    ----------
    cell_distance: float
        Distance between the center of two cells.
    number_cells_per_side: int
        How many cells are on each side (so that for an Angel Lobster the number of cells would be number_cells_per_side**2, 
        but not in this case)
    cell_side: float
        Lenght of the side of each cell.
    cell_length: float
        Mirror length of each cell.
    radius: float
        Curvature radius of the KB layer.
    f: float
        Focal length of the KB layer.
    """
    cell_centers, sides = cell_centers_sides_equal_adaptiveL(cell_distance, number_cells_per_side, cell_side)
    L_top = L_top_compute(radius, cell_length, cell_centers)
    L_bottom = L_bottom_adaptiveL(f, L_top, cell_centers, sides, radius)
    return cell_centers, sides, L_bottom, L_top

def generate_KB_cells_adaptive_side(cell_distance, number_cells_per_side, cell_side, cell_length, radius, f):
    """
    Function to generate the cells of a KB with the side of cells adaptive so we have a complete filling factor. 
    Note, we call each aperture of the KB cell, in analogy with an Angel Lobster, even if this design is different.

    Parameters
    ----------
    cell_distance: float
        Distance between the center of two cells.
    number_cells_per_side: int
        How many cells are on each side (so that for an Angel Lobster the number of cells would be number_cells_per_side**2, 
        but not in this case)
    cell_side: float
        Lenght of the side of each cell.
    cell_length: float
        Mirror length of each cell.
    radius: float
        Curvature radius of the KB layer.
    f: float
        Focal length of the KB layer.
    """
    cell_centers, sides = cell_centers_sides_adaptiveside(cell_distance, number_cells_per_side, cell_side, f, radius, cell_length)
    L_top = L_top_compute(radius, cell_length, cell_centers)
    L_bottom = np.repeat(radius, cell_centers.size)
    return cell_centers, sides, L_bottom, L_top


def define_KB_mirrors(configuration, cell_distance, number_cells_per_side, cell_side, cell_length, radius, radius_ort=None, 
                      cells_arrengement_function=generate_KB_cells_equal, complete_vectorization=False, max_reflections=2):
    """
    Defines the geometrical parameters of a KB layer of mirrors. Note, we call each aperture of the KB cell, in analogy with an 
    Angel Lobster, even if this design is different.

    Parameters
    ----------
    configuration: str
        Can be either 'KB_top', 'KB_bottom according to which layer one wants to generate.
    ell_distance: float
        Distance between the center of two cells.
    number_cells_per_side: int
        How many cells are on each side (so that for an Angel Lobster the number of cells would be number_cells_per_side**2, 
        but not in this case)
    cell_side: float
        Lenght of the side of each cell.
    cell_length: float
        Mirror length of each cell.
    radius: float
        Curvature radius of the KB layer.
    radius_ort: float
        Curvature radius of the other KB layer.
    cells_arrengement_function: function
        Function generating the parameters of the cells.
    complete_vectorization: bool
        Experimental, if True, the numpy calculations are vectorized (but in this case each 'cell' can only hit the one in front).
    max_reflections: int
        Maximum number of reflections simulated.
    """

    assert configuration in ['KB_top', 'KB_bottom']
    assert not complete_vectorization
    
    if configuration == 'KB_top':
        f = radius_ort/2
        cardinal_keys = ['nord', 'sud']
    elif configuration in ['KB_bottom']:
        assert radius_ort is None
        f = radius/2
        cardinal_keys = ['est', 'ovest']
    
    #we call them cells even if they are not really, because this way it's named similar to Lobster
    cell_centers, sides, L_bottom, L_top = cells_arrengement_function(cell_distance, number_cells_per_side, cell_side, cell_length, radius, f) 
    
    side_single = np.repeat(number_cells_per_side*cell_distance, cell_centers.size)
    long_side_max = side_single/2
    long_side_min = -side_single/2
    
    def generate_surface_exists():
        return np.repeat(True, cell_centers.size)
    def determine_nonreflecting_sides(surface_key):
        surface_exists = generate_surface_exists()
        if surface_key in ['nord', 'est']:
            surface_exists[cell_centers <= 0] = False
        elif surface_key in ['sud', 'ovest']:
            surface_exists[cell_centers >= 0] = False
        return surface_exists
    
    
    surface_is_reflecting = {}
    for cardinal_key in cardinal_keys:
        surface_is_reflecting[cardinal_key] = determine_nonreflecting_sides(cardinal_key)
    
    dict_tot = {}
    dict_tot['f'] = f
    dict_tot['cell_centers'] = cell_centers
    dict_tot['sides'] = sides
    dict_tot['L_bottom'] = L_bottom
    dict_tot['L_top'] = L_top
    dict_tot['long_side_min'] = long_side_min
    dict_tot['long_side_max'] = long_side_max
    dict_tot['radius'] = radius
    dict_tot['reflecting'] = surface_is_reflecting
    dict_tot['max_reflections'] = max_reflections
    dict_tot['n_configs'] = cell_centers.size
    return dict_tot


def create_KB_mirrors(configuration, telescope_pars):
    """
    Generates the python functions to describe a KB telescope layer (either the top or the ortogonal bottom one).

    Parameters
    ----------
    configuration: str
        Can be either 'KB_top', 'KB_bottom' according to which layer one wants to generate.
    telescope_pars: dict
        Geometrical parameters of the KB telescope layer, obtained with the define_KB_mirrors function.
    """

    f = telescope_pars['f']
    cell_centers = telescope_pars['cell_centers'] 
    sides = telescope_pars['sides']
    L_bottom = telescope_pars['L_bottom']
    L_top = telescope_pars['L_top']
    long_side_min = telescope_pars['long_side_min']
    long_side_max = telescope_pars['long_side_max']
    surface_is_reflecting = telescope_pars['reflecting'] 
    
    bounds_inf_KB = [-np.inf, np.inf]
    bounds_KB  = [long_side_min, long_side_max]
    cardinal_mirrors = {}
    if configuration == 'KB_top':
        cardinal_mirrors['nord'] = KB_parabola_dataclass(f=f, L=L_top, R0=cell_centers + sides/2, axis='y', z_low=L_bottom, z_up=L_top, bounds_x=bounds_KB, bounds_y=bounds_inf_KB)
        cardinal_mirrors['sud']  = KB_parabola_dataclass(f=f, L=L_top, R0=cell_centers - sides/2, axis='y', z_low=L_bottom, z_up=L_top, bounds_x=bounds_KB, bounds_y=bounds_inf_KB)  
        sides_x = long_side_max - long_side_min
        sides_y = sides
        x_at_top = np.repeat(0, cell_centers.size)
        y_at_top = cell_centers
    elif configuration in ['KB_bottom']:
        cardinal_mirrors['est']   = KB_parabola_dataclass(f=f, L=L_top, R0=cell_centers + sides/2, axis='x', z_low=L_bottom, z_up=L_top, bounds_x=bounds_inf_KB, bounds_y=bounds_KB)
        cardinal_mirrors['ovest'] = KB_parabola_dataclass(f=f, L=L_top, R0=cell_centers - sides/2, axis='x', z_low=L_bottom, z_up=L_top, bounds_x=bounds_inf_KB, bounds_y=bounds_KB)
        sides_x = sides 
        sides_y = long_side_max - long_side_min
        x_at_top = cell_centers
        y_at_top = np.repeat(0, cell_centers.size)

    from Lobster_ray_tracing import Lobster_telescope_array_geometry
    
    return Lobster_telescope_array_geometry(sides_x=sides_x, sides_y=sides_y, 
                                            x_at_top=x_at_top, y_at_top=y_at_top, 
                                            z_up=L_top,
                                            cardinal_mirrors=cardinal_mirrors, 
                                            surface_is_reflecting=surface_is_reflecting)
    
