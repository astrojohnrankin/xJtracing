import numpy as np
 
# TEMPORARY
# import os, sys
# sys.path.append(os.path.dirname(__file__))
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append("..")
from xJtracing.rays import create_ray, create_ray, rotate_ray_dataclass
from xJtracing.tracing_utils import assert_single_number, assert_array_1d

def create_circular_corona_of_rays(angle_off_axis, 
                                   rmin, rmax,
                                   z0, rays_in_mm2=10, energy=1, distribute_in_field_radius=False, pa_angle=0):
    """
    Generates rays randomly distribued in a circular corona.

    Parameters
    ----------
    angle_off_axis: float
        Polar angle of source.
    pa_angle: float
        Position angle of source.
    rmin, rmax: array of float
        Radii of each circular corona.
    z0: float
        z coordinate of where rays start, should coincide with upper aperture of telescope.
    rays_in_mm2: float
        Density of rays.
    energy: float
        Energy, currently can only be monochromatic.
    distribute_in_field_radius: float
        if not False, angle_off_axis is ignored and rays are distributed over a circular region of radius distribute_in_field_radius.

    Notes
    -----
    Position angles are sampled from a uniform distribution, to assure that all points are also uniformly distributed in spherical geometry,
    the radial coordinate is sampled from $\sqrt{random(r_{min}^2, r_{max}^2)}$.
    """
    
    Area_proj = np.pi*(rmax**2 - rmin**2) #1d [n_configurations]
    n_rays = (rays_in_mm2*Area_proj).astype('int') #1d [n_configurations]
    n_configs = rmin.size #float
    rays_array_shape = [n_rays.max(), n_configs]
    rmin_2d = np.tile(rmin, [n_rays.max(), 1]) #2d [n_rays_max, n_configurations]?
    rmax_2d = np.tile(rmax, [n_rays.max(), 1]) #2d [n_rays_max, n_configurations]?
    
    r_random = np.sqrt(np.random.uniform(rmin_2d**2, rmax_2d**2, rays_array_shape)) #2d [n_rays_max, n_configurations]
    alpha_random = np.random.uniform(0, 2*np.pi, rays_array_shape)
    x_randoms = r_random*np.cos(alpha_random)
    y_randoms = r_random*np.sin(alpha_random)

    if distribute_in_field_radius: #mi sa che questa parte è sbagliata
        rho =  np.random.triangular(0, distribute_in_field_radius, distribute_in_field_radius, rays_array_shape)
        theta = np.random.uniform(0, 2*np.pi, rays_array_shape)
    else:
        rho = np.tile(angle_off_axis, rays_array_shape)
        theta = np.tile(pa_angle, rays_array_shape)
    
    rays = create_ray(rho, theta, 
                    x_randoms, y_randoms, z0=np.tile(z0, [rays_array_shape[0], 1]), 
					energy=energy,
                    area_over_Nrays = 1/rays_in_mm2)
    for i, _n_rays in enumerate(n_rays):
        rays.delete_ray[_n_rays+1:, i] += 1  #the matrix of rays has to be simmetric, but the outer shells have more rays            
    return rays


def create_circular_corona_of_rays_for_WI(telescope_pars, **kwargs):
    """
    Creates a circular corona of rays, with the minimum radius corresponding to the intersection plane (on-axis),
    and the outer radius to the outer radius of the parabola. See create_circular_corona_of_rays for the parameters.
    """

    return create_circular_corona_of_rays(rmin=telescope_pars['radii_center'], 
                                          rmax = telescope_pars['radii_parabola'], 
                                          z0=telescope_pars['L1s'], **kwargs)


# def create_rays_ground_tube(angle_off_axis, D, R0, mirror_inc_angle, L1, f0, rays_in_mm2=10, energy=1):
# LA PARTE DELLA DISTRIBUZIONE TRIANGOLARE E' SBAGLIATA
#     """
#     Generates rays in a circular corona, originating from a finite distance source, such as PANTER.

#     Parameters
#     ----------
#     angle_off_axis: float
#         Polar angle of source, which in an actual facility corresponds to a rotation of the source.
#     D: float
#         Distance to source
#     R0, mirror_inc_angle, L1: array of float
#         Radii, inclination and length of each circular corona's mirrors
#     f0: float
#         focal distance of optics
#     rays_in_mm2: float
#         Density of rays
#     energy: float
#         Energy, currently can only be monochromatic
#     """
#     rmin = R0 - 2 #radius in mm
#     rmax = R0 + L1*np.tan(mirror_inc_angle) + 2
#     r_min = np.arctan(rmin/D) #radial angle
#     r_max = np.arctan(rmax/D)
#     Area_proj = np.pi*(rmax**2 - rmin**2)
#     n_rays = (rays_in_mm2*Area_proj).astype('int')
#     n_configs = R0.size #float
    
#     rays_array_shape = [n_rays.max(), n_configs]
#     rmin_1d = np.tile(r_min, [n_rays.max(), 1]) #2d [n_rays_max, n_configurations]
#     rmax_2d = np.tile(r_max, [n_rays.max(), 1]) #2d [n_rays_max, n_configurations]
    
#     r_random = np.random.triangular(rmin_1d, rmax_2d, rmax_2d, rays_array_shape)
#     alpha_random = np.random.uniform(0, 2*np.pi, rays_array_shape)
        
#     rays_original = create_ray(r_random, alpha_random, 
#                                 x0=np.tile(D*np.arctan(angle_off_axis), rays_array_shape), 
#                                 energy=energy,
#                                 y0=np.zeros(rays_array_shape), z0=np.tile(D, rays_array_shape), area_over_Nrays = 1/rays_in_mm2)
#     rays = rotate_ray_dataclass(rays_original, angle_off_axis)
#     for i, _n_rays in enumerate(n_rays):
#         rays.delete_ray[_n_rays+1:, i] += 1  
#     fp = 1/(1/f0 - 1/D)
#     return rays, fp


# def create_ground_rays_for_WI(telescope_pars, **kwargs):
#     """
#     Creates ground rays for a Wolter-I with telescope parameters telescope_pars, with source not positioned at infinite 
#     but at 130m, like in PANTER facility.
#     See create_rays_ground_tube for the parameters.
#     NOT TESTED YET! 
#     remember: need to modify input to optimize focal plane chosen to fp = 1/(1/f0 - 1/D) !!!
#     """
#     f0s = np.repeat(telescope_pars['f0'], telescope_pars['radii_center'].size)
#     theta = np.arctan2(telescope_pars['radii_center'], f0s)/4
#     return create_rays_ground_tube(angle_off_axis, D=130e3, R0=telescope_pars['radii_center'], mirror_inc_angle=theta, 
#                                    L1=telescope_pars['L1s'], f0=f0s, **kwargs)


def create_square_rays(angle_off_axis, position_angle, sides_x, sides_y, x_at_top, y_at_top, z_up, rays_in_mm2=50, energy=1):
    """
    Generates rays distributed in a grid of squares or rectangles, useful for Lobsters and similar.

    Parameters
    ----------
    angle_off_axis: float
        Polar angle of source.
    position_angle: float
        Position angle of source.
    sides_x, sides_y: np.ndarray
        Vector of the sides of the squares or rectangles in which to generate the rays.
    x_at_top, y_at_top: np.ndarray
        Coordinates of the top center of each square or rectangle.
    z_up: np.ndarray
        Height of the top of each square or rectangle.
    rays_in_mm2: float
        Density of rays.
    energy: float
        Energy, currently can only be monochromatic.
    """
    assert_single_number(angle_off_axis, position_angle)
    assert_single_number(rays_in_mm2, energy)
    assert_array_1d(sides_x, sides_y, x_at_top, y_at_top, z_up)
    
    Area_proj = (sides_x*sides_y)
    n_rays = np.around(rays_in_mm2*Area_proj).astype('int')
    n_configs = sides_x.size
    rays_array_shape = [n_rays.max(), n_configs]
    rho = np.tile(angle_off_axis, rays_array_shape)
    theta = np.tile(position_angle, rays_array_shape)
    x_random = np.random.uniform(x_at_top - sides_x/2, x_at_top + sides_x/2, rays_array_shape)
    y_random = np.random.uniform(y_at_top - sides_y/2, y_at_top + sides_y/2, rays_array_shape)
        
    rays = create_ray(rho=rho, 
                      theta=theta, 
                      x0=x_random, y0=y_random, z0=np.tile(z_up, [rays_array_shape[0], 1]), 
                      energy=energy,
                      area_over_Nrays=1/rays_in_mm2,
                      geometric_area = n_rays/rays_in_mm2)

    for i, _n_rays in enumerate(n_rays):
        rays.delete_ray[_n_rays:, i] += 1

    return rays


def create_square_rays_for_Angel(lobster_array, **kwargs):
    """
    Creates an array of rays in a grid, over the apertures of the lobster array of an Angel lobster.
    """
    return create_square_rays(sides_x=lobster_array.sides_x, sides_y=lobster_array.sides_y, 
                              x_at_top=lobster_array.x_at_top, y_at_top=lobster_array.y_at_top, z_up=lobster_array.z_up, **kwargs)


def create_square_rays_for_Schmidt(telescope_top, telescope_bottom, complete_vectorization, **kwargs):
    """
    Creates an array of rays in a grid, over the apertures of the lobster arrays (both top and bottom) of a Schmidt lobster.

    Parameters
    ----------
    telescope_top, telescope_bottom: dictionary
       Parameters of the mirrors.
    complete_vectorization: bool
        Experimental, if True, the numpy calculations are vectorized (but in this case each 'cell' can only hit the one in front).
    """
    if complete_vectorization:
        print('telescope_top.sides_x', telescope_top.sides_x, telescope_bottom.sides_x)
        print('telescope_top.x_at_top', telescope_top.x_at_top, telescope_bottom.x_at_top)
  
    return create_square_rays(sides_x=telescope_bottom.sides_x if complete_vectorization else telescope_top.sides_x,
                              sides_y=telescope_top.sides_y, 
                              x_at_top=telescope_bottom.x_at_top if complete_vectorization else telescope_top.x_at_top,
                              y_at_top=telescope_top.y_at_top, z_up=telescope_top.z_up, **kwargs)

      