import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# TEMPORARY
import os, sys
# sys.path.append(os.path.dirname(__file__))
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append("..")

from xJtracing.mirror import flat_mirror_dataclass
from xJtracing.rays import reflect_ray_starting_from_mirror, mark_rays_for_non_deletion, delete_rays_marked_for_deletion, generate_ray_direction_equations
from xJtracing.intersection import create_image
from xJtracing.benchmarks import all_hew_info_from_cross
from xJtracing.generators import create_square_rays_for_Angel, create_square_rays_for_Schmidt
from xJtracing.KB import create_KB_mirrors
from xJtracing.plotting import plot_mirror, plot_rays, plot_rays_flattened


@dataclass(frozen=True)
class Lobster_telescope_array_geometry:
   """
   Lobster_telescope_array_geometry
   """
   sides_x: float
   sides_y: float
   x_at_top: float
   y_at_top: float
   z_up: float
   cardinal_mirrors: float
   surface_is_reflecting: bool


def Lobster_derived_geometrical_parameters(alpha, theta, x0, y0, L_high, radius):
    """
    Projects the lobster geometrical parameters over the x and y axes. 

    Notes
    -----
    See define_Lobster_mirrors's documentation for the equations derivation.
    """
    alpha_x = np.arctan(np.tan(alpha)*np.cos(theta))
    alpha_y = np.arctan(np.tan(alpha)*np.sin(theta))
    x_at_top = x0 + (L_high + radius)*np.sin(alpha)*np.cos(theta)
    y_at_top = y0 + (L_high + radius)*np.sin(alpha)*np.sin(theta)
    return alpha_x, alpha_y, x_at_top, y_at_top
  

def create_Lobster_mirrors(configuration, telescope_pars):
    """
    Generates the python functions to describe an Angel lobster telescope or a Schmidt Lobster layer 
    (either the top or the ortogonal bottom one).

    Parameters
    ----------
    configuration: str
        Can be either 'Angel', 'Schmidt_top', 'Schmidt_bottom'  according to which layer one wants to generate.
    telescope_pars: dict
        Geometrical parameters of the telescope or telescoe layer, obtained with the define_Lobster_mirrors function.

    Notes
    -----
    See define_Lobster_mirrors's documentation for the equations derivation, like for the boundaries.
    """
    
    x0 = telescope_pars['x0']
    y0 = telescope_pars['y0']
    surface_is_reflecting = telescope_pars['surface_is_reflecting']

    alpha_x, alpha_y, x_at_top, y_at_top = Lobster_derived_geometrical_parameters(telescope_pars['alpha'], telescope_pars['theta'], x0, y0, telescope_pars['L_high'], telescope_pars['radius_curvature'])

    nord_sud_configurations = ['Angel', 'Schmidt_top']
    est_ovest_configurations = ['Angel', 'Schmidt_bottom']

    if configuration == 'Schmidt_top':
        z_offset = telescope_pars['radius_curvature'] - telescope_pars['radius']
    else:
        z_offset = 0
    
    bounds_z = [(telescope_pars['radius_curvature']+telescope_pars['L_low'])*np.cos(telescope_pars['alpha']) - z_offset, (telescope_pars['radius_curvature']+telescope_pars['L_high'])*np.cos(telescope_pars['alpha']) - z_offset] 
    if configuration in 'Angel':
        bounds_nord_sud  = lambda z: [x0 - telescope_pars['sides_x']/2 + z*np.tan(alpha_x), x0 + telescope_pars['sides_x']/2 + z*np.tan(alpha_x)]
        bounds_est_ovest = lambda z: [y0 - telescope_pars['sides_y']/2 + z*np.tan(alpha_y), y0 + telescope_pars['sides_y']/2 + z*np.tan(alpha_y)]
    elif configuration in ['Schmidt_top', 'Schmidt_bottom']:
        bounds_nord_sud  = lambda z: [x0 - telescope_pars['side_long']/2, x0 + telescope_pars['side_long']/2]
        bounds_est_ovest = lambda z: [y0 - telescope_pars['side_long']/2, y0 + telescope_pars['side_long']/2]
    bounds_inf = lambda z: [-np.inf, np.inf]
    
    theta_pi_2 = np.tile(np.pi/2, x_at_top.shape)
    theta_0 = np.tile(0, x_at_top.shape)
    x0s = np.tile(x0, x_at_top.shape)
    y0s = np.tile(y0, x_at_top.shape)
    z0s = np.tile(-z_offset, x_at_top.shape)

    cardinal_mirrors = {}
    if configuration in nord_sud_configurations:
        cardinal_mirrors['nord'] = flat_mirror_dataclass(alpha=alpha_y, theta=theta_pi_2, x0=x0s, y0=y0s + telescope_pars['sides_y']/2, z0=z0s, z_low=bounds_z[0], z_up=bounds_z[1], bounds_x=bounds_nord_sud, bounds_y=bounds_inf)
        cardinal_mirrors['sud']  = flat_mirror_dataclass(alpha=alpha_y, theta=theta_pi_2, x0=x0s, y0=y0s - telescope_pars['sides_y']/2, z0=z0s, z_low=bounds_z[0], z_up=bounds_z[1], bounds_x=bounds_nord_sud, bounds_y=bounds_inf)
    if configuration in est_ovest_configurations: #non elif!!!
        cardinal_mirrors['est']   = flat_mirror_dataclass(alpha=alpha_x, theta=theta_0   , x0=x0s + telescope_pars['sides_x']/2, y0=y0s, z0=z0s, z_low=bounds_z[0], z_up=bounds_z[1], bounds_x=bounds_inf, bounds_y=bounds_est_ovest)
        cardinal_mirrors['ovest'] = flat_mirror_dataclass(alpha=alpha_x, theta=theta_0   , x0=x0s - telescope_pars['sides_x']/2, y0=y0s, z0=z0s, z_low=bounds_z[0], z_up=bounds_z[1], bounds_x=bounds_inf, bounds_y=bounds_est_ovest)
    
    if telescope_pars['reflecting']=='total':
        surface_is_reflecting = {}
        def generate_surface_exists():
            return np.repeat(True, x_at_top.shape)
        for cardinal_key in telescope_pars['cardinal_keys']:
            surface_is_reflecting[cardinal_key] = generate_surface_exists()
        

    return Lobster_telescope_array_geometry(sides_x=telescope_pars['sides_x'], sides_y=telescope_pars['sides_y'], 
                                            x_at_top=x_at_top, y_at_top=y_at_top, z_up=bounds_z[1],
                                            cardinal_mirrors=cardinal_mirrors, 
                                            surface_is_reflecting=surface_is_reflecting)

    
    
def simulate_lobster_rays(rays, cardinal_mirrors, surface_is_reflecting, material_nk_files_list, d_list, rugosity, 
                          max_reflections=2, configuration='Angel', ax_x=None, ax_y=None):
    """
    Simulates the passage of rays inside a cellular optics module (either a full Angel Lobster telescope or a Schmidt or KB layer).

    Parameters
    ----------
    rays: instance of rays_dataclass
        Input rays, vectorized for the different cells.
    cardinal_mirrors: dictionary
        Dictionary of the mirrors functions, generated for example with the create_Lobster_mirrors function or similars.
    surface_is_reflecting: dictionary
        Boolean values indicating if a certain surface is reflecting.
    material_nk_files_list: list
        List of table files containing refraction index real and imaginary components as a function 
        of energy. If it is None a refractive index of 1, such as for air, is assumed.
    d_list: list
        List of thicknesses of the mirrors.
    rugosity: float
        Rugosity of the mirrors causes scattering in random directions, quantified by this value (in arcsec), and degrading the hew.
    max_reflections: int
        Maximum number of reflections simulated.
    configuration: str
        Can be either 'Angel', 'Schmidt_top', 'Schmidt_bottom'  according to which layer one wants to simulate.
    
    Notes
    -----
    The logical flow of this computation is the following:
    - find stray rays that don't hit any surface
    - choose a surface
        - filter rays that don't hit one of the other 3
            - try hitting on that surface
                - try reflecting the output on other 3
                    - goes to detector (no hit other 3)
                    - reflected (repeat by trying hitting on other 3 --- which are now different)
                        - stop this passage when no more rays left
    
    """
    assert configuration in ['Angel', 'Schmidt_top', 'Schmidt_bottom', 'KB_top', 'KB_bottom']
    if configuration=='Angel':
        cardinal_keys = ['nord', 'sud', 'est', 'ovest']
        other_surface_dict = {
                    'nord':  ['sud', 'est', 'ovest'],
                    'sud':   ['est', 'ovest', 'nord'],
                    'est':   ['ovest', 'nord', 'sud'],
                    'ovest': ['nord', 'sud', 'est']
                    }
    elif configuration in ['Schmidt_top', 'KB_top']:
        cardinal_keys = ['nord', 'sud']
        other_surface_dict = {'nord': ['sud'], 'sud': ['nord']}
    elif configuration in ['Schmidt_bottom', 'KB_bottom']:
        cardinal_keys = ['est', 'ovest']
        other_surface_dict = {'est': ['ovest'], 'ovest': ['est']}

    if configuration in ['KB_top', 'KB_bottom']:
        mirror_is_already_flat = False
    else:
        mirror_is_already_flat = True
        
    def _reflect_rays_on_surface(input_rays, surface, surface_is_reflecting_):
        reflected_rays, exists_intersection = reflect_ray_starting_from_mirror(ray_original=input_rays, mirror=surface, 
                                    material_nk_files_list=material_nk_files_list, d_list=d_list, 
                                                                               mirror_is_already_flat=mirror_is_already_flat, rugosity=rugosity,
                                                                              ax_x=ax_x, ax_y=ax_y, plot_type_x='x Angel', plot_type_y='y Angel')
                                
        reflected_rays_selected = mark_rays_for_non_deletion(reflected_rays, np.logical_and(exists_intersection, surface_is_reflecting_ ) )
        non_reflected_rays_condition = ~exists_intersection
        return reflected_rays_selected, non_reflected_rays_condition
        
    failed_all_reflections_condition = []
    dict_5 = {}
    for cardinal_key in cardinal_keys:
        reflected_rays_selected, non_reflected_rays_condition = _reflect_rays_on_surface(rays, cardinal_mirrors[cardinal_key], surface_is_reflecting[cardinal_key])
        failed_all_reflections_condition.append(non_reflected_rays_condition)
        dict_5[cardinal_key] = [reflected_rays_selected]
    dict_5['to_detector'] = [[mark_rays_for_non_deletion(rays, np.logical_and.reduce(failed_all_reflections_condition))]]
    
    def recursive_multiple_reflections(dict_5):    
        return_dict = dict(map(lambda _key: [_key, []], cardinal_keys))
        number_rays_reflected = 0
        to_detector_rays = []
        
        for _key in cardinal_keys:
            for input_rays in dict_5[_key]:
                to_detector_rays_conditions = []
                for other_surface_key in other_surface_dict[_key]:
                    reflected_rays_selected, non_reflected_rays_condition = _reflect_rays_on_surface(input_rays, cardinal_mirrors[other_surface_key], surface_is_reflecting[other_surface_key])
                    to_detector_rays_conditions.append(non_reflected_rays_condition)
                    return_dict[other_surface_key].append(reflected_rays_selected)
                    number_rays_reflected += reflected_rays_selected.e[0].size
                to_detector_rays.append(mark_rays_for_non_deletion(input_rays, np.logical_and.reduce(to_detector_rays_conditions)))
        
        return_dict['to_detector'] = [*dict_5['to_detector'], to_detector_rays]
        
        if number_rays_reflected==0 or len(return_dict['to_detector']) > max_reflections:
            return return_dict['to_detector']
        else:
            return recursive_multiple_reflections(return_dict)
        
    return recursive_multiple_reflections(dict_5)
    
    
def simulate_lobster_tube(lobster_tube, material_nk_files_list, d_list, rugosity, rays, focal_plane, max_reflections=2, ax_x=None, ax_y=None):
    """
    Simulates the ray tracing inside an Angel lobster system and crates the image.

    Parameters
    ----------
    lobster_tube: instance of Lobster_telescope_array_geometry
        The input telescope.
    material_nk_files_list: list
        List of table files containing refraction index real and imaginary components as a function 
        of energy. If it is None a refractive index of 1, such as for air, is assumed.
    d_list: list
        List of thicknesses of the mirrors.
    rugosity: float
        Rugosity of the mirrors causes scattering in random directions, quantified by this value (in arcsec), and degrading the hew.
    rays: instance of rays_dataclass
        Input rays, vectorized for the different cells.
    max_reflections: int
        Maximum number of reflections simulated.
    
    Note: at the moment the focal plane is flat
    """    
    reflected_list = simulate_lobster_rays(rays=rays, cardinal_mirrors=lobster_tube.cardinal_mirrors, 
                                           surface_is_reflecting=lobster_tube.surface_is_reflecting, 
                                           material_nk_files_list=material_nk_files_list, 
                                           d_list=d_list, rugosity=rugosity, max_reflections=max_reflections, ax_x=ax_x, ax_y=ax_y)
    
    def create_lobster_image_(i):
        rays_dict = {_key:np.array([]) for _key in ['x', 'y', 'A_eff', 'N_initial', 'N_final']}
        for reflected_rays in reflected_list[i]:
            reflected_rays_passed = mark_rays_for_non_deletion(reflected_rays, reflected_rays.survival==True)
            rays_final = delete_rays_marked_for_deletion(reflected_rays_passed)
            x_det, y_det, z_det = create_image(rays_final.e, rays_final.x0, -focal_plane)
            rays_dict['x'] = np.append(rays_dict['x'], x_det)
            rays_dict['y'] = np.append(rays_dict['y'], y_det) 
            rays_dict['N_initial'] = np.append(rays_dict['N_initial'], reflected_rays.e[0].size) 
            rays_dict['N_final'] = np.append(rays_dict['N_final'], rays_final.e[0].size) 
            rays_dict['A_eff'] = np.append(rays_dict['A_eff'], rays_final.area_over_Nrays*rays_final.e[0].size)
            rays_dict['area_over_Nrays'] = rays_final.area_over_Nrays

            if ax_x is not None: 
                plot_rays(ax_x, reflected_rays, focal_plane, reflected_rays.x0[2], side='x')
            if ax_y is not None: 
                plot_rays(ax_y, reflected_rays, focal_plane, reflected_rays.x0[2], side='y')
        return i, rays_dict
    
    rays_maps = dict(map(create_lobster_image_, range(len(reflected_list))))

    #queste righe successive si potrebbero compattare inteliggentemente
    # if ax_x is not None or ax_y is not None:
    #     for rays_maps_n in rays_maps:
    #         for rays_to_plot in rays_maps[rays_maps_n]:
    #             if ax_x is not None: 
    #                 plot_rays(ax_x, rays_maps[rays_maps_n][rays_to_plot], -focal_plane, rays_maps[rays_maps_n][rays_to_plot].x0[2])
    #             if ax_y is not None:
    #                 plot_rays(ax_y, rays_maps[rays_maps_n][rays_to_plot], -focal_plane, rays_maps[rays_maps_n][rays_to_plot].x0[2])
    if ax_x is not None: 
        ax_x.axhline(focal_plane, color='green', lw=4)
    if ax_y is not None:
        ax_y.axhline(focal_plane, color='green', lw=4)
    
    #total maps of rays:
    x_combined = np.array([])
    y_combined = np.array([])
    A_eff = 0
    for i in rays_maps.keys():
        # if i in [0, 1]:
        if i in [0, 1, 2, 3, 4]:
            x_combined = np.append(x_combined, rays_maps[i]['x'])
            y_combined = np.append(y_combined, rays_maps[i]['y'])
        if i in [1, 2]:
            A_eff += rays_maps[i]['A_eff'].sum()

    rays_maps['total'] = {'x':x_combined, 'y':y_combined, 'A_eff':A_eff}
    return rays_maps


def simulate_an_Angel(off_axis_angle_deg, pa_deg, telescope_pars, rays_function=create_square_rays_for_Angel, 
                      energy=1, rays_in_mm2=50,
                      material_nk_files_list = ['data/nk/Au.nk'], d_list = [], rugosity_hew = False,
                        finite_distance_source = False, distribute_in_field_radius=False, plot_tracing=False):
    """
    Full simulation of an Angel Lobster system.

    Parameters
    ----------
    angle_off_axis_deg: float
        Polar angle of source [in deg].
    pa_deg: float
        Position angle of source [in deg].
    telescope_pars: dict
        Geometrical parameters of the telescope, generated using the define_Lobster_mirrors function.
    rays_function: function
        Function to generate the input rays with the chosen distribution.
    energy: float
        Energy, currently can only be monochromatic.
    rays_in_mm2: float
        Density of rays.
    material_nk_files_list: list
        List of table files containing refraction index real and imaginary components as a function 
        of energy. If it is None a refractive index of 1, such as for air, is assumed.
    d_list: list
        List of thicknesses of the mirrors.
    rugosity_hew: float
        Rugosity of the mirrors causes scattering in random directions, quantified by this value (in arcsec), and degrading the hew.
    finite_distance_source: bool
        Not currently implemented, would compute a finite distance source such as PANTER facility.
    distribute_in_field_radius: float
        if not False, angle_off_axis is ignored and rays are distributed over a circular region of radius distribute_in_field_radius.
    plot_tracing: bool
        If True, ray tracing is plotted.
    """
    assert finite_distance_source == False, 'not implemented otherwise yet'

    lobster_array = create_Lobster_mirrors(configuration='Angel', telescope_pars=telescope_pars)
    
    rays = rays_function(angle_off_axis=off_axis_angle_deg*np.pi/180, position_angle=pa_deg*np.pi/180, 
                              lobster_array=lobster_array,
                              rays_in_mm2=rays_in_mm2, energy=energy)

    #count initial rays
    rays_for_counting = delete_rays_marked_for_deletion(rays)
    nrays_initial = rays_for_counting.e[0].size

    if plot_tracing is not False:
        if plot_tracing is True: 
            fig, [ax_up, ax_low] = plt.subplots(1, 2, sharex=True, sharey=True)
            fig.tight_layout()
        else:
            ax_up, ax_low = plot_tracing
        ax_up.set_xlabel('x [mm]')
        ax_low.set_xlabel('y [mm]')
        ax_up.set_ylabel('z [mm]')
        
        plot_mirror(ax_low, lobster_array.cardinal_mirrors['nord'], side='y Lobster')
        plot_mirror(ax_low, lobster_array.cardinal_mirrors['sud'], side='y Lobster')
        plot_mirror(ax_up, lobster_array.cardinal_mirrors['est'], side='x Lobster')
        plot_mirror(ax_up, lobster_array.cardinal_mirrors['ovest'], side='x Lobster')

    else:
        ax_up, ax_low = None, None

    rays_maps = simulate_lobster_tube(lobster_tube=lobster_array, 
                                    material_nk_files_list=material_nk_files_list, 
                                      d_list=d_list, 
                                    rugosity=rugosity_hew, rays=rays, 
                                      focal_plane=telescope_pars['radius']/2, 
                                      max_reflections=telescope_pars['max_reflections'], ax_x=ax_low, ax_y=ax_up)



    dict_return = {}
    dict_return['Aeff'] = rays_maps['total']['A_eff']
    dict_return['x'] = rays_maps['total']['x']
    dict_return['y'] = rays_maps['total']['y']
    dict_return['rays_maps'] = rays_maps
    dict_return['rays_N_return'] = {_n_refl:{_N_key:rays_maps[_n_refl][_N_key][0] for _N_key in ['N_initial', 'N_final']} for _n_refl in range(telescope_pars['max_reflections']+1)}
    dict_return['hew_dict'] = all_hew_info_from_cross(rays_maps, radius=telescope_pars['radius'])
    dict_return['hew'] = dict_return['hew_dict']['hew']
    dict_return['nrays_initial'] = nrays_initial

    return dict_return


# Ortogonals
# ================================================================


def simulate_ortogonal_system(telescope_top, telescope_bottom, top_configuration, bottom_configuration, material_nk_files_list, d_list, rugosity, rays, focal_plane,
                             max_reflections_top, max_reflections_bottom, complete_vectorization, ax_x=None, ax_y=None):
    """
    Simulates the rays inside a telescope system made of two ortogonal layers of mirrors: Schmidt Lobster or KB.

    Parameters
    ----------
    telescope_top, telescope_bottom: dict
        Geometrical parameters of the telescopes, generated using the define_Lobster_mirrors function.
    top_configuration, bottom_configuration: str
        Can be either 'Schmidt_top', 'Schmidt_bottom' or 'KB_top', 'KB_bottom' according to what one wants to simulate.
    material_nk_files_list: list
        List of table files containing refraction index real and imaginary components as a function 
        of energy. If it is None a refractive index of 1, such as for air, is assumed.
    d_list: list
        List of thicknesses of the mirrors.
    rugosity: float
        Rugosity of the mirrors causes scattering in random directions, quantified by this value (in arcsec), and degrading the hew.
    rays: instance of rays_dataclass
        Input rays, vectorized for the different cells.
    focal_plane: float
        Distance to the focal plane from the bottom mirrors.
    max_reflections_top, max_reflections_bottom: int
        Maximum number of reflections simulated.
    complete_vectorization: bool
        Experimental, if True, the numpy calculations are vectorized (but in this case each 'cell' can only hit the one in front).
        
    """
    # top surface reflections
    # =======================
    dict_3 = simulate_lobster_rays(rays=rays, cardinal_mirrors=telescope_top.cardinal_mirrors, 
                                           surface_is_reflecting=telescope_top.surface_is_reflecting, 
                                           material_nk_files_list=material_nk_files_list, 
                                           d_list=d_list, rugosity=rugosity, max_reflections=max_reflections_top,
                                           configuration=top_configuration, ax_x=ax_x, ax_y=ax_y)
    
    # bottom surface reflections
    # ==========================
    final_rays_lists = []
    rays_reflected_n = {0:[], '1up':[], '1down':[], 2:[], '3+':[]}
    for n_top, subset_rays_n in enumerate(dict_3):
        for subset_rays in subset_rays_n:

            if complete_vectorization:
                dict_3_bottom_i = simulate_lobster_rays(rays=subset_rays, cardinal_mirrors=telescope_bottom.cardinal_mirrors, 
                                               surface_is_reflecting=telescope_bottom.surface_is_reflecting, 
                                               material_nk_files_list=material_nk_files_list, 
                                               d_list=d_list, rugosity=rugosity, max_reflections=max_reflections_bottom,
                                              configuration=bottom_configuration, ax_x=ax_x, ax_y=ax_y)
        
                for n_bottom, subset_bottom_rays_n in enumerate(dict_3_bottom_i):
                    for subset_bottom_rays in subset_bottom_rays_n:
                        final_rays_lists.append(subset_bottom_rays)
                        if n_top==0 and n_bottom==0:
                            rays_reflected_n[0].append(subset_bottom_rays)
                        elif n_top==1 and n_bottom==0:
                            rays_reflected_n['1up'].append(subset_bottom_rays)
                        elif n_top==0 and n_bottom==1:
                            rays_reflected_n['1down'].append(subset_bottom_rays)
                        elif n_top+n_bottom==2:
                            rays_reflected_n[2].append(subset_bottom_rays)
                        elif n_top+n_bottom>2:
                            rays_reflected_n['3+'].append(subset_bottom_rays)
            else:
                if ax_x:
                    plot_rays(ax_x, subset_rays, z_low=subset_rays.x0[2], z_up=subset_rays.x0[2], side='x')
                if ax_y:
                    plot_rays(ax_y, subset_rays, z_low=subset_rays.x0[2], z_up=subset_rays.x0[2], side='y')
                
                arrived_rays = delete_rays_marked_for_deletion(subset_rays)
                # arrived_rays = subset_rays

                for surface_i in range(len(telescope_bottom)):
                    
                    x_func, y_func = generate_ray_direction_equations(arrived_rays.e, arrived_rays.x0)
                    x_rays = x_func(telescope_bottom[surface_i].cardinal_mirrors['est'].z_up)

                    xmin = telescope_bottom[surface_i].x_at_top - telescope_bottom[surface_i].sides_x/2
                    xmax = telescope_bottom[surface_i].x_at_top + telescope_bottom[surface_i].sides_x/2
                    rays_selected_xrange = mark_rays_for_non_deletion(arrived_rays, np.logical_and(x_rays > xmin[0], 
                                                                                                   x_rays < xmax[0]))

                    
        
                    dict_3_bottom_i = simulate_lobster_rays(rays=rays_selected_xrange, cardinal_mirrors=telescope_bottom[surface_i].cardinal_mirrors, 
                                               surface_is_reflecting=telescope_bottom[surface_i].surface_is_reflecting,
                                                            material_nk_files_list=material_nk_files_list, d_list=d_list, rugosity=rugosity, 
                               max_reflections=max_reflections_bottom, configuration=bottom_configuration)
    
                    for n_bottom, subset_bottom_rays_n in enumerate(dict_3_bottom_i):
                        for subset_bottom_rays in subset_bottom_rays_n:
                            final_rays_lists.append(subset_bottom_rays)
    
                            if n_top==0 and n_bottom==0:
                                rays_reflected_n[0].append(subset_bottom_rays)
                            elif n_top==1 and n_bottom==0:
                                rays_reflected_n['1up'].append(subset_bottom_rays)
                            elif n_top==0 and n_bottom==1:
                                rays_reflected_n['1down'].append(subset_bottom_rays)
                            elif n_top+n_bottom==2:
                                rays_reflected_n[2].append(subset_bottom_rays)
                            elif n_top+n_bottom>2:
                                rays_reflected_n['3+'].append(subset_bottom_rays)

    # to detector
    # ===========
    def _create_image(reflected_rays):
        reflected_rays_passed = mark_rays_for_non_deletion(reflected_rays, reflected_rays.survival==True)
        rays_final = delete_rays_marked_for_deletion(reflected_rays_passed)
        x_det, y_det, z_det = create_image(rays_final.e, rays_final.x0, -focal_plane)            
        return x_det, y_det, rays_final.e[0].size
        
    dict_tot = {'total':{_key:np.array([]) for _key in ['x', 'y', 'A_eff']}}
    for reflected_rays in final_rays_lists:
        x_det, y_det, rays_final = _create_image(reflected_rays)
        dict_tot['total']['x'] = np.append(dict_tot['total']['x'], x_det)
        dict_tot['total']['y'] = np.append(dict_tot['total']['y'], y_det)
        
    
    rays_reflected_n_dict = {}
    for n_key in rays_reflected_n.keys():
        rays_reflected_n_dict[n_key] = {_key:np.array([]) for _key in ['x', 'y', 'N_initial', 'N_final']}
        for reflected_rays in rays_reflected_n[n_key]:
            if ax_x is not None: 
                # import pdb; pdb.set_trace()
                plot_rays_flattened(ax_x, reflected_rays, focal_plane, reflected_rays.x0[2], side='x')
            if ax_y is not None: 
                plot_rays_flattened(ax_y, reflected_rays, focal_plane, reflected_rays.x0[2], side='y')
            
            x_det, y_det, rays_final = _create_image(reflected_rays)
            rays_reflected_n_dict[n_key]['x'] = np.append(rays_reflected_n_dict[n_key]['x'], x_det)
            rays_reflected_n_dict[n_key]['y'] = np.append(rays_reflected_n_dict[n_key]['y'], y_det)
            rays_reflected_n_dict[n_key]['N_initial'] = np.append(rays_reflected_n_dict[n_key]['N_initial'], reflected_rays.e[0].size) 
            rays_reflected_n_dict[n_key]['N_final'] = np.append(rays_reflected_n_dict[n_key]['N_final'], rays_final) 
            rays_reflected_n_dict[n_key]['area_over_Nrays'] = reflected_rays.area_over_Nrays
    
    dict_tot['A_eff'] = rays.area_over_Nrays*(rays_reflected_n_dict['1up']['x'].size + rays_reflected_n_dict['1down']['x'].size + rays_reflected_n_dict[2]['x'].size) #+ rays_reflected_n_dict['3+']['x'].size )
    for _key in rays_reflected_n_dict.keys():
        dict_tot[_key] = rays_reflected_n_dict[_key]

    return dict_tot

def convert_telescope_pars_to_list(telescope_pars_bottom):
    """
    Converts a dictionary of lists to a list of dictionary of 1d lists.
    """
    telescope_pars_bottom_new = []
    for i in range(telescope_pars_bottom['n_configs']):
        def convert_item(label):
            if isinstance(telescope_pars_bottom[label], np.ndarray):
                return label, np.array([telescope_pars_bottom[label][i]])
            elif isinstance(telescope_pars_bottom[label], dict):
                def sub_item_f(label2):
                    return label2, np.array([telescope_pars_bottom[label][label2][i]])
                return label, dict(map(sub_item_f, list(telescope_pars_bottom[label].keys())))
            else:
                return label, telescope_pars_bottom[label]
        single_dict = dict(map(convert_item, list(telescope_pars_bottom.keys())))
        telescope_pars_bottom_new.append(single_dict)
    return telescope_pars_bottom_new
    

def simulate_a_Schmidt_or_KB(off_axis_angle_deg, pa_deg, telescope_pars_top, telescope_pars_bottom, rays_function=create_square_rays_for_Schmidt, 
                       energy=1, rays_in_mm2=3,
                       material_nk_files_list = ['data/nk/Au.nk'], d_list = [], rugosity_hew = False,
                       finite_distance_source = False, distribute_in_field_radius=False, design='Schmidt', complete_vectorization=False, plot_tracing=False):
    """
    Full simulation of a Schmidt or KB Lobster system.

    Parameters
    ----------
    angle_off_axis_deg: float
        Polar angle of source [in deg].
    pa_deg: float
        Position angle of source [in deg].
    telescope_pars_top, telescope_pars_bottom: dict
        Geometrical parameters of the telescope, generated using the define_Lobster_mirrors function.
    rays_function: function
        Function to generate the input rays with the chosen distribution.
    energy: float
        Energy, currently can only be monochromatic.
    rays_in_mm2: float
        Density of rays.
    material_nk_files_list: list
        List of table files containing refraction index real and imaginary components as a function 
        of energy. If it is None a refractive index of 1, such as for air, is assumed.
    d_list: list
        List of thicknesses of the mirrors.
    rugosity_hew: float
        Rugosity of the mirrors causes scattering in random directions, quantified by this value (in arcsec), and degrading the hew.
    finite_distance_source: bool
        Not currently implemented, would compute a finite distance source such as PANTER facility.
    distribute_in_field_radius: float
        if not False, angle_off_axis is ignored and rays are distributed over a circular region of radius distribute_in_field_radius.
    design: str
        Can be either 'Schmidt' or 'KB'.
    complete_vectorization: bool
        Experimental, if True, the numpy calculations are vectorized (but in this case each 'cell' can only hit the one in front).
    plot_tracing: bool
        If True, ray tracing is plotted.
    """

    assert design in ['Schmidt', 'KB']

    if design=='Schmidt':
        top_configuration = 'Schmidt_top'
        bottom_configuration = 'Schmidt_bottom'
        mirror_function = create_Lobster_mirrors
    elif design=='KB':
        top_configuration = 'KB_top'
        bottom_configuration = 'KB_bottom'
        mirror_function = create_KB_mirrors
        
    telescope_top = mirror_function(top_configuration, telescope_pars_top)
    telescope_bottom_vector = mirror_function(bottom_configuration, telescope_pars_bottom)
    if complete_vectorization:
        telescope_bottom = telescope_bottom_vector 
    else:
        telescope_pars_bottom_list = convert_telescope_pars_to_list(telescope_pars_bottom)
        telescope_bottom = list(map(lambda i: mirror_function(bottom_configuration, telescope_pars_bottom_list[i]), range(len(telescope_pars_bottom_list))))

    rays = rays_function(angle_off_axis=off_axis_angle_deg*np.pi/180, position_angle=pa_deg*np.pi/180, 
                                  telescope_top=telescope_top, telescope_bottom=telescope_bottom, complete_vectorization=complete_vectorization,
                                  rays_in_mm2=rays_in_mm2, energy=energy)

    #count initial rays
    rays_for_counting = delete_rays_marked_for_deletion(rays)
    nrays_initial = rays_for_counting.e[0].size

    if plot_tracing is not False:
        if plot_tracing is True: 
            fig, [ax_up, ax_low] = plt.subplots(1, 2, sharex=True, sharey=True)
            fig.tight_layout()
        else:
            ax_up, ax_low = plot_tracing
        ax_up.set_xlabel('x [mm]')
        ax_low.set_xlabel('y [mm]')
        ax_up.set_ylabel('z [mm]')
        # import pdb; pdb.set_trace()
        plot_mirror(ax_low, telescope_top.cardinal_mirrors['nord'], side='y')
        plot_mirror(ax_low, telescope_top.cardinal_mirrors['sud'], side='y')
        # import pdb; pdb.set_trace()
        plot_mirror(ax_up, telescope_bottom_vector.cardinal_mirrors['est'], side='x')
        plot_mirror(ax_up, telescope_bottom_vector.cardinal_mirrors['ovest'], side='x')
    
    else:
        ax_up, ax_low = None, None

    rays_maps = simulate_ortogonal_system(telescope_top, telescope_bottom, top_configuration, bottom_configuration, material_nk_files_list,
                                          d_list, rugosity_hew, rays, telescope_pars_bottom['radius']/2, 
                                          telescope_pars_top['max_reflections'], telescope_pars_bottom['max_reflections'], complete_vectorization, ax_x=ax_low, ax_y=ax_up)

    dict_return = {}
    dict_return['x'] = rays_maps['total']['x']
    dict_return['y'] = rays_maps['total']['y']
    dict_return['Aeff'] = rays_maps['A_eff']
    dict_return['rays_maps'] = rays_maps
    dict_return['hew_dict'] = all_hew_info_from_cross(rays_maps, radius=telescope_pars_bottom['radius'],
                                  # hew_method='only center')
                                    # hew_method='use cross'
                                                     )
    dict_return['hew'] = dict_return['hew_dict']['hew']
    dict_return['nrays_initial'] = nrays_initial
    return dict_return
    
