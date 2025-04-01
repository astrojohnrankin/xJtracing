import numpy as np
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
from dataclasses import replace as dc_replace
import copy
import pandas as pd

# TEMPORARY
import os, sys
# sys.path.append(os.path.dirname(__file__))
from xJtracing.reflections import find_normal_to_flat_plane, find_tangent_plane_and_normal, reflect_ray, angle_two_vectors, rotate_normal_for_rugosity, rotate_ray
from xJtracing.absorption import generate_Fresnel_coefficients, reflection_R_many_layers
from xJtracing.plotting import plot_rays

def ray_versor(rho, theta):
    """
    Versor indicating a direction given polar coordinates. 
    If input is a vector multiple rays can be simulated.
    
    Parameters
    ----------
    rho: array_like
        Radial angle [rad]
    theta: array_like
        Position angle [rad]
    Returns
    -------
    (e1, e2, e3): tuple of 3 array_like
        Base describing the versor.
        
    Notes
    -----
    While light is here approximated as rays, it can also be described as waves, 
    from Maxwell's equations the wave equation in three dimensions is found
    
    .. math::
		\\rot^2 \\vec E = \\mu \\epsilon \\frac{\\partial^2 \\vec E}{\\partial t^2}
		
    Looking for solutions to this equation of the form (separating temporal and
    spatial parts)
    
    .. math::
    	\\vec E(\\vec r, t) = \\hat n \\tilde E(\\vec r)e^{\\jmath \\omega t}
    	
    one finds Helmhotz equation
    
    .. math::
    	(\\rot^2 + k^2)\\tilde E (\\vec r)
    where :math:`k = \\omega / c` is the wave number.
    
    Every point in the source can be considered to be emitting spherical waves according
    to Huygens principle.
    
    Paraxial approximation: all wavefront normals make small angles with the z-axis.
    Writing the :math:`\\tilde E (x,y,z)= u(x,y,z)e^{-\\jmath k z}`, so that the phase change in z
    is separate, and assuming that :math:`\\gradient u / \\gradient z` is small compared to 
    :math:`\\lambda`, the Helmhotz equation, can be written as
    
    .. math::
		\\frac{\\partial^2 u}{\\partial x^2} + \\frac{\\partial^2 u}{\\partial y^2} - 2\\jmath k \\frac{\\partial u}{\\partial z} = 0
    
    Temporal coherence: all Huygens wavelets are emitting at same frequency, same phase
    for a long time

    Spatial coherence: all Huygens emitters are in phase one with another.
    """
    e1 = np.sin(rho)*np.cos(theta)
    e2 = np.sin(rho)*np.sin(theta)
    e3 = np.cos(rho)
    return np.array([e1, e2, e3])


def generate_ray_direction_equations(e, x_0):
    """
    Equation of a line given a point and its direction's polar coordinates.
    
    Parameters
    ----------
    e: tuple of 3 array_like
        Base e1, e2, e3 describing the versor.
    x_0: tuple of 3 array_like
        A point (x0, y0, z0) through which the array passes.
        
    Notes
    -----
    A ray is defined as 
    
    .. math::
    	\\vec x = \\vec x_0 + \\vec e * t
    
    If input is a vector multiple rays can be simulated.
    """    
    e1, e2, e3 = e
    x0, y0, z0 = x_0
    def x_func(z):
        t = (z-z0)/e3
        return x0 + t*e1
    def y_func(z):
        t = (z-z0)/e3
        return y0 + t*e2
    return x_func, y_func
  


# Data structures to organize the above definitions:
# ==================================================
  
    
@dataclass(frozen=True)
class rays_dataclass:
   """
   Container for all rays properties, see xJtracing.rays
   """ 
   e: tuple
   x0: tuple
   energy: float
   survival: float
   delete_ray: float
   area_over_Nrays: float
   geometric_area: float
    

def create_ray(rho, theta, x0, y0, z0, energy, area_over_Nrays=None, geometric_area=None):
    """
    Creates a ray starting from its angles and a point it passes through.
    See xJtracing.rays for all definitions of parameters.

    Parameters
    ----------
    rho: float
        Polar angle of source
    theta: float
        Position angle of source
    rmin, rmax: array of float
        Radii of each circular corona
    x0, y0z0: float
        Where rays start, z coordinate should coincide with upper aperture of telescope
    rays_in_mm2: float
        Density of rays
    energy: float
        Energy, currently can only be monochromatic
    area_over_Nrays, geometric_area: float
        Information used to computer benchmarks later
    """
    e = ray_versor(rho, theta)
    survival = np.tile(True, rho.shape) # Before interacting the ray has passed (maybe through empty space or something)
    delete_ray = np.zeros(rho.shape, dtype=np.uint8) # An array which can be used in the end to mask rays that we select for all the many reasons
    return rays_dataclass(e=e, x0=np.array([x0, y0, z0]), energy=energy, survival=survival, 
                                   delete_ray=delete_ray, area_over_Nrays=area_over_Nrays, geometric_area=geometric_area)
    

def find_if_ray_was_reflected(inc_angle, energy, material_nk_file): 
    """
    Finds if ray was reflected by random probability. Valid for one material interfaced
    with empty space. For the parameters see xJtracing.absorption.
    """
    coeffs = generate_Fresnel_coefficients(None, material_nk_file)(inc_angle, energy)
    return np.random.random_sample(inc_angle.shape) < (np.abs(coeffs.Rs)**2 + np.abs(coeffs.Rp)**2)/2


def find_if_ray_was_reflected_many_layers(inc_angle, energy, d_list, material_nk_files_list): 
    """
    Finds if ray was reflected by random probability. Valid for many materials. 
    For the parameters see xJtracing.absorption.
    """   
    coeffs = reflection_R_many_layers(inc_angle, energy, d_list, material_nk_files_list)
    return np.random.random_sample(inc_angle.shape) < (np.abs(coeffs.Rs)**2 + np.abs(coeffs.Rp)**2)/2 
    

def create_multilayer_lookup_table(energy, d_list, material_nk_files_list):
    """
    Creates a table of absorption parameters to speed computations when doing multilayer.
    For the parameters see xJtracing.absorption.
    """
    inc_angles_arange = np.arange(-np.pi/2, np.pi/2, 0.005*np.pi/180)
    coeffs = reflection_R_many_layers(inc_angles_arange, energy=energy, 
                                                  d_list=d_list, material_nk_files_list=material_nk_files_list)
    R = (np.abs(coeffs.Rs)**2 + np.abs(coeffs.Rp)**2)/2
    def lookup_table_function(input_angles):
        return np.random.random_sample(input_angles.shape) < np.interp(input_angles, inc_angles_arange, R, left=np.nan, right=np.nan)
    
    return ['use lookup table', lookup_table_function]


def reflect_ray_starting_from_mirror(ray_original, mirror, material_nk_files_list=[], d_list=[], mirror_is_already_flat=False, rugosity=False, 
                                     ax_x=None, ax_y=None, plot_type_x='radial', plot_type_y='radial'):
    """
    Reflects the ray. Absorption is considered if material_nk_file is not None.
    If mirror is a flat mirror, the mirror_is_already_flat flag should be set to True to
    avoid the derivatives, etc.
    If rugosity is not None, a scattering due to rugose surface is considered, with half
    energy width equal to the value given to rugosity [rad].
    For the parameters see xJtracing.rays.
    ax_x, ax_y: axis
        Matplotlib axes. If not False it plots the rays on the two axes.
    """
    ray = copy.deepcopy(ray_original)
    (x_intersect, y_intersect, z_intersect), exists_intersection = mirror.intersection_function(x_0=ray.x0, e=ray.e)
    if mirror_is_already_flat:
        n_normal = find_normal_to_flat_plane(mirror.alpha, mirror.theta, x_intersect, y_intersect, z_intersect)
    else:
        tan_plane_z, n_normal = find_tangent_plane_and_normal(mirror.equation, x_intersect, y_intersect)
    
    if rugosity:
        n_normal = rotate_normal_for_rugosity(rugosity, np.array(n_normal), np.array(ray.e))
    
    e_out = reflect_ray(ray.e, n_normal, x_intersect, y_intersect, z_intersect)
    incidence_angle = angle_two_vectors(e_out, n_normal)
    
    #3 conditions basically: multilayer, one layer and no absorption (no need to write code for that)
    if len(material_nk_files_list) >= 1:
        if material_nk_files_list[0] == 'use lookup table':
            lookup_table = material_nk_files_list[1]
            _ray_was_reflected = lookup_table(incidence_angle)
        elif len(material_nk_files_list) > 1:
            _ray_was_reflected = find_if_ray_was_reflected_many_layers(incidence_angle, ray.energy, d_list, material_nk_files_list)  
        elif len(material_nk_files_list) == 1:   
            _ray_was_reflected = find_if_ray_was_reflected(incidence_angle, ray.energy, material_nk_files_list[0])   
        survival = np.where(_ray_was_reflected==0, False, ray.survival) 
    else:
        survival = ray.survival

    x_new = np.where(exists_intersection, x_intersect, ray_original.x0[0])
    y_new = np.where(exists_intersection, y_intersect, ray_original.x0[1])
    z_new = np.where(exists_intersection, z_intersect, ray_original.x0[2])
    new_rays = dc_replace(ray, e=e_out, x0=np.array([x_new, y_new, z_new]), survival=survival)


    if ax_x:
        plot_rays(ax_x, ray_original, z_low=new_rays.x0[2], z_up=ray_original.x0[2], side=plot_type_x)
    if ax_y:
        plot_rays(ax_y, ray_original, z_low=new_rays.x0[2], z_up=ray_original.x0[2], side=plot_type_y)

    return new_rays, exists_intersection


def mark_rays_for_non_deletion(ray_original, mask):
    """
    Select rays based on mask
    The rays need to be selected at the very end using the ray_original.delete_ray mask, which is >0 
    where the rays are to be deleted, 0 if they are to be saved.
    """
    ray = copy.deepcopy(ray_original)
    delete_ray = np.where(mask, ray.delete_ray, ray.delete_ray+1)
    return dc_replace(ray, delete_ray=delete_ray)


def delete_rays_marked_for_deletion(ray_original):
    """
    Deletes rays that have been marked for deletion by the mark_rays_for_non_deletion function
    """
    ray = copy.deepcopy(ray_original)
    mask = np.where(ray.delete_ray==0, True, False)
    return dc_replace(ray, e=np.array([_e[mask] for _e in ray.e]), 
                          x0=np.array([_x0[mask] for _x0 in ray.x0]), 
                    survival=ray.survival[mask], delete_ray=ray.delete_ray[mask])


def rotate_ray_dataclass(ray_original, theta):
    """Rotate ray around y of angle theta [rad]
    DEPRECATED
    """
    ray = copy.deepcopy(ray_original)
    e_rot = rotate_ray(ray.e, theta)
    return dc_replace(ray, e=e_rot)


def tilt_rays_vector(rays_e, tild_deg, tilt_deg_pa, inverse=False):
    """
    Tilt the rays, changing this way the reference system.

    Parameters
    ----------
    rays_e: array_like
        A 3 dimensional array of shape [3, number of rays, number of shells]; it is the direction versor of the rays.
    tild_deg: float, or np.ndarray of size equal to n_configs
        Tilt angle, in degrees.
    tilt_deg_pa: float, or np.ndarray of size equal to n_configs
        Position angle of the tilt, in degrees.
    inverse: bool
        If True, the inverse rotation is applied. Useful when returning to original reference system.
    """
    rays_e = copy.deepcopy(rays_e)
    assert len(rays_e.shape) == 3
    if not isinstance(tild_deg, np.ndarray): tild_deg = np.repeat(tild_deg, rays_e.shape[2])
    if not isinstance(tilt_deg_pa, np.ndarray): tilt_deg_pa = np.repeat(tilt_deg_pa, rays_e.shape[2])
    def rotate_config(config_idx):
        rotation_ = Rotation.from_rotvec([-np.sin(tilt_deg_pa[config_idx]*np.pi/180)*tild_deg[config_idx], 
                                          np.cos(tilt_deg_pa[config_idx]*np.pi/180)*tild_deg[config_idx], 0], degrees=True)
        e_ = rays_e[:,:,config_idx].T
        return rotation_.apply(e_, inverse=inverse)
    return np.array(list(map(rotate_config, range(rays_e.shape[2])))).T


def tilt_rays(rays, tild_deg, tilt_deg_pa, inverse=False, z_rotation=None, xshift=None, yshift=None):
    """
    Tilt the rays, changing this way the reference system.

    Parameters
    ----------
    rays: instance of rays_dataclass
        A 3 dimensional array of shape [3, number of rays, number of shells]; it is the direction versor of the rays.
    tild_deg: float, or np.ndarray of size equal to n_configs
        Tilt angle, in degrees.
    tilt_deg_pa: float, or np.ndarray of size equal to n_configs
        Position angle of the tilt, in degrees.
    inverse: bool
        If True, the inverse rotation is applied. Useful when returning to original reference system.
    z_rotation: float
        Distance around which to rotate the rays. If it is None, the rays are rotated around the z already present in rays.x0.
    xshift: np.ndarray of size equal to n_configs
        shift of rays on x
    yshift: np.ndarray of size equal to n_configs
        shift of rays on y
    """
    rotated_e = tilt_rays_vector(rays.e, tild_deg, tilt_deg_pa, inverse=inverse)
    if z_rotation is not None:
        
        # 1. faccio passare per -L
        pre_ray_direction_x, pre_ray_direction_y = generate_ray_direction_equations(rays.e, rays.x0)
        x0_bottom = pre_ray_direction_x(z_rotation)
        y0_bottom = pre_ray_direction_y(z_rotation)
        z0_bottom = np.repeat(z_rotation[np.newaxis, :], x0_bottom.shape[0], axis=0)
        x_0_bottom = np.array([x0_bottom, y0_bottom, z0_bottom])
        
        # 2. gli x0 di sopra vengono ruotati con tilt_rays_vector!!! con inverso
        x_0_bottom_rot = tilt_rays_vector(x_0_bottom, tild_deg, tilt_deg_pa, inverse=inverse)
    
        # 3. i raggi ruotati vengon fatti passare per i punti trovari sopra
        x_det, y_det, z_det = x_0_bottom_rot[0], x_0_bottom_rot[1], x_0_bottom_rot[2]

        
    #     #raggi in uscita a 
    #     pre_ray_direction_x, pre_ray_direction_y = generate_ray_direction_equations(rays.e, rays.x0)
    #     x0_bottom = pre_ray_direction_x(exit_z)
    #     y0_bottom = pre_ray_direction_y(exit_z)
    #     # import pdb; pdb.set_trace()
    #     z0_bottom = np.repeat(exit_z[np.newaxis, :], x0_bottom.shape[0], axis=0)
    #     # import pdb; pdb.set_trace()
    #     ray_direction_x, ray_direction_y = generate_ray_direction_equations(rotated_e, np.array([x0_bottom, y0_bottom, z0_bottom]))
    #     x_det, y_det = ray_direction_x(z_rotation), ray_direction_y(z_rotation)
    #     # z_det = np.tile(z_rotation, x_det.shape)
    #     z_det = np.repeat(z_rotation[np.newaxis, :], x_det.shape[0], axis=0)
    else:
        x_det, y_det, z_det = rays.x0[0], rays.x0[1], rays.x0[2]
        
    if xshift is not None:
        x_det -= xshift
    if yshift is not None:
        y_det -= yshift

    x_0_new = np.array([x_det, y_det, z_det])
    return dc_replace(rays, e=rotated_e, x0=x_0_new)


def save_rays_to_file(rays, savepath):
    """
    Saves rays direction and 0 point to a file in various formats.

    Parameters
    ----------
    rays: instance of rays_dataclass
        The rays to save; the absorbed and rays marked for deletion are not saved.
    savepath: str
        Path where to save file (not including extension).
    """
    rays2 = mark_rays_for_non_deletion(rays, rays.survival==True)
    rays3 = delete_rays_marked_for_deletion(rays2)
    df = pd.DataFrame({'e1':rays3.e[0], 'e2':rays3.e[1], 'e3':rays3.e[2],
                 'x0':rays3.x0[0], 'y0':rays3.x0[1], 'z0':rays3.x0[2]})
    df.to_csv(savepath+'.csv', index=False)
    df.to_csv(savepath+'.txt', sep=' ', index=False)
    df.to_excel(savepath+'.xlsx', index=False)
    