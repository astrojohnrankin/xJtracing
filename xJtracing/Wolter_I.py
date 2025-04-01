import numpy as np
import matplotlib.pyplot as plt
 
# TEMPORARY
import os, sys
# sys.path.append(os.path.dirname(__file__))
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append("..")
from xJtracing.rays import reflect_ray_starting_from_mirror, mark_rays_for_non_deletion, delete_rays_marked_for_deletion, tilt_rays, save_rays_to_file
from xJtracing.generators import create_circular_corona_of_rays_for_WI
from xJtracing.intersection import create_image, find_best_focal_plane
from xJtracing.tracing_utils import assert_array_1d
from xJtracing.mirror import parabola_dataclass, iperbola_dataclass
from xJtracing.benchmarks import half_energy_diameter
from xJtracing.plotting import plot_mirror, plot_rays


def simulate_rays_Wolter_I(rays, material_nk_files_list, d_list, parabola, iperbola, inner_mirror, parabola_inner, iperbola_inner, rugosity, ax=None):
    """
    Simulate rays inside a Wolter-I

    Parameters
    ----------
    rays: instance of rays_dataclass
        Input rays
    material_nk_files_list: list
        Materials of which mirrors are made
    d_list: list
        Thicknesses of mirror layers
    parabola: instance of parabola_dataclass
        Parabolas
    iperbola: instance of iperbola_dataclass
        Iperbolas
    inner_mirror: bool
        If true, the simulation considers absorption and reflections from inner mirror
    parabola_inner, iperbola_inner: instances of parabola_dataclass and iperbola_dataclass
        Inner parabolas and iperbolas
    rugosity: float
        If not 0 or False, rays are randomly scattered (of 2 times rugosity) to account for imperfectly polished surface.
    ax: axis
        Matplotlib axis. If not False it plots the rays on ax.
    """
    
    def _reflect_ray(rays, surface):
        reflected_ray_all, exists_intersection = reflect_ray_starting_from_mirror(rays, surface, 
                                            material_nk_files_list=material_nk_files_list, 
                                            d_list=d_list, mirror_is_already_flat=False, rugosity=rugosity, ax_x=ax)
        reflected_ray = mark_rays_for_non_deletion(reflected_ray_all, exists_intersection)
        non_reflected_ray = mark_rays_for_non_deletion(rays, ~exists_intersection) 
        return reflected_ray, non_reflected_ray
    
    if inner_mirror: #Simulation considering absorption and reflections from inner mirror
    
        def test_ray_for_absorption(rays):
            reflected_ray_all_inner_parabola, exists_intersection_inner_parabola = reflect_ray_starting_from_mirror(rays, parabola_inner, 
                                    material_nk_files_list=material_nk_files_list, d_list=d_list, rugosity=rugosity)
            reflected_ray_all_inner_iperbola, exists_intersection_inner_iperbola = reflect_ray_starting_from_mirror(rays, iperbola_inner, 
                                    material_nk_files_list=material_nk_files_list, d_list=d_list, rugosity=rugosity)
            rays_survived_from_absorption = mark_rays_for_non_deletion(rays,
                                                np.logical_and(~exists_intersection_inner_parabola,
                                                ~exists_intersection_inner_iperbola))
            return rays_survived_from_absorption
        
        #first we try to reflect on parabola
        rays_survived_I_absorption = test_ray_for_absorption(rays)
        reflected_ray_parabola, non_reflected_ray_parabola = _reflect_ray(rays_survived_I_absorption, parabola)
            
        # if it fails I try to reflect on iperbola
        non_reflected_ray_parabola_survived_inner = test_ray_for_absorption(non_reflected_ray_parabola)
        reflected_ray_only_iperbola_, failed_all_reflections_ = _reflect_ray(non_reflected_ray_parabola_survived_inner, iperbola)  
        reflected_ray_only_iperbola = test_ray_for_absorption(reflected_ray_only_iperbola_)
        failed_all_reflections = test_ray_for_absorption(failed_all_reflections_)
        
        # if I reflection does not fail I try to reflect also on iperbola
        reflected_ray_parabola_survived_inner = test_ray_for_absorption(reflected_ray_parabola)
        double_reflected_, reflected_only_parabola_ = _reflect_ray(reflected_ray_parabola_survived_inner, iperbola)
        double_reflected = test_ray_for_absorption(double_reflected_)
        reflected_only_parabola = test_ray_for_absorption(reflected_only_parabola_)
         
    else:
        #first we try to reflect on parabola
        reflected_ray_parabola, non_reflected_ray_parabola = _reflect_ray(rays, parabola)
            
        # if it fails I try to reflect on iperbola
        reflected_ray_only_iperbola, failed_all_reflections = _reflect_ray(non_reflected_ray_parabola, iperbola)    
        
        # if I reflection does not fail I try to reflect also on iperbola
        double_reflected, reflected_only_parabola = _reflect_ray(reflected_ray_parabola, iperbola)

                                    
    return failed_all_reflections, reflected_ray_only_iperbola, reflected_only_parabola, double_reflected
   
        
        
def simulate_a_WolterI(off_axis_angle_deg, pa_deg, telescope_pars, rays_function=create_circular_corona_of_rays_for_WI, energy=1, rays_in_mm2=50,
                       material_nk_files_list = ['data/nk/Au.nk'], d_list = [], rugosity_hew = False,
                       optimize_focal_plane=False, plot_tracing=False):
    """
    Function performing a complete simulations of a Wolter-I multi-shell telescope defined by parameters inside telescope_pars.

    Parameters
    ----------
    off_axis_angle_deg: float
        Polar angle (in deg) of source
    pa_deg: float
        Position angle (in deg) of source
    telescope_pars: dictionary
        radii_parabola: array of float
            External (maximum) radii of parabolas
        radii_center: array of float
            Radii at intersection of parabolas and iperbolas
        radii_center_inner: array of float
            Radii at intersection of inner parabolas and iperbolas (so differes from radii_center by the thickness)
        L1s: array of float
            Lenght of each parabola and iperbola (so total mirror lenght is 2*L)
        f0: float
            Focal lenght of telescope
        inner_mirror: bool
            If true, the simulation considers absorption and reflections from inner mirrors
        apply_tilt: bool or dict
        If not False, it is a dict with parameters tilt_deg and pa_deg that tilt the entire telescope.
    rays_function: function
        Function returning the rays.
    energy: float
        Energy, currently can only be monochromatic
    rays_in_mm2: float
        Density of rays
    material_nk_files_list: list
        Materials of which mirrors are made
    d_list: list
        Thicknesses of mirror layers
    rugosity_hew: float
        If not 0 or False, rays are randomly scattered (of 2 times rugosity_hew) to account for imperfectly polished surface.
    optimize_focal_plane: bool
        Finds the focal plane that minizes the hew for the given off_axis_angle_deg.
    plot_tracing: bool
        If True, ray tracing is plotted.

    Notes
    -----
    The angles of the parabola and iperbola :math:`\\alpha` and :math:`\\theta` are not given in input, inside telescope_pars, they are instead derived in this function to achieve the
    maximum reflection efficiency:
    
    .. math::
    	\\theta = \\arctan{\\frac{R}{f_0}}/4
    	\\beta = 3\\theta
    """

    if 'apply_tilt' in telescope_pars:
        apply_tilt = telescope_pars['apply_tilt']
    else:
        apply_tilt = False
    
    f0s = np.repeat(telescope_pars['f0'], telescope_pars['radii_center'].size)
    theta = np.arctan2(telescope_pars['radii_center'], f0s)/4
    beta = 3*theta

    image_dict_tot = {}

    if rugosity_hew: 
        rugosity_keys = ['liscio', 'rugoso']
        rugosities = [False, rugosity_hew]
    else:
        rugosity_keys = ['liscio']
        rugosities = [False]
        
    for rugosity_key, rugosity in zip(rugosity_keys, rugosities):
        image_dict_tot[rugosity_key] = {}
        assert_array_1d(f0s, telescope_pars['radii_center'], telescope_pars['L1s'], theta, beta)  
        
        parabola = parabola_dataclass(R0=telescope_pars['radii_center'], theta=theta, z_low = np.zeros(telescope_pars['L1s'].shape), z_up = telescope_pars['L1s'])
        iperbola = iperbola_dataclass(R0=telescope_pars['radii_center'], beta=beta, theta=theta, f0=f0s, z_low = -telescope_pars['L1s'], z_up = np.zeros(telescope_pars['L1s'].shape))
        if telescope_pars['inner_mirror']:
            parabola_inner = parabola_dataclass(R0=telescope_pars['radii_center_inner'], theta=theta, z_low = np.zeros(telescope_pars['L1s'].shape), z_up = telescope_pars['L1s'])
            iperbola_inner = iperbola_dataclass(R0=telescope_pars['radii_center_inner'], beta=beta, theta=theta, f0=f0s, z_low = -telescope_pars['L1s'], z_up = np.zeros(telescope_pars['L1s'].shape))
        else:
            parabola_inner, iperbola_inner = None, None
        
        rays = rays_function(angle_off_axis=off_axis_angle_deg*np.pi/180, telescope_pars=telescope_pars, 
                                                  rays_in_mm2=rays_in_mm2, energy=energy, pa_angle=pa_deg*np.pi/180)

        #count initial rays
        rays_for_counting = delete_rays_marked_for_deletion(rays)
        nrays_initial = rays_for_counting.e[0].size


        if plot_tracing is not False:
            if plot_tracing is True: 
                fig, ax = plt.subplots()
            else:
                ax = plot_tracing
            ax.set_xlabel('x [mm]')
            ax.set_ylabel('z [mm]')

            plot_mirror(ax, parabola)
            plot_mirror(ax, iperbola)
            if telescope_pars['inner_mirror']:
                plot_mirror(ax, parabola_inner, inner=True)
                plot_mirror(ax, iperbola_inner, inner=True)

        else:
            ax=None

        if apply_tilt:
            rays = tilt_rays(rays, apply_tilt['tilt_deg'], apply_tilt['pa_deg'])

        failed_all_reflections, reflected_ray_only_iperbola, reflected_only_parabola, double_reflected = simulate_rays_Wolter_I(rays=rays, 
                                                                        material_nk_files_list=material_nk_files_list, d_list=d_list, parabola=parabola, iperbola=iperbola, 
                                                                        inner_mirror=telescope_pars['inner_mirror'], parabola_inner=parabola_inner, iperbola_inner=iperbola_inner, rugosity=rugosity, ax=ax)

        if plot_tracing is not False:
            # for rays_to_plot in [reflected_ray_only_iperbola]:
            for rays_to_plot in [failed_all_reflections, reflected_ray_only_iperbola, reflected_only_parabola, double_reflected]:
                plot_rays(ax, rays_to_plot, -telescope_pars['f0'], rays_to_plot.x0[2])


        def create_image_rays_of_type_(reflected_rays, label, savepath=None):
            rays_dict = {}
            reflected_rays_passed = mark_rays_for_non_deletion(reflected_rays, reflected_rays.survival==True)
            rays_final = delete_rays_marked_for_deletion(reflected_rays_passed)

            if savepath is not None:
                save_rays_to_file(rays_final, savepath)
            
            if optimize_focal_plane is True:
                best_focal_plane = find_best_focal_plane(rays_final, telescope_pars['f0'], 200)
            else:
                best_focal_plane = telescope_pars['best_focal_plane']
            
            x_det, y_det, z_det = create_image(rays_final.e, rays_final.x0, best_focal_plane)
            rays_dict['x'] = x_det
            rays_dict['y'] = y_det
            rays_dict['A_eff'] = rays_final.area_over_Nrays*rays_final.e[0].size #dovrebbe essere solo di questi specifici raggi riflessi N volte
            rays_dict['best_focal_plane'] = best_focal_plane
            return label, rays_dict

        # create_image_rays_of_type_(double_reflected, "Double reflected", '/Users/John/Desktop/nontilted')
        
        if apply_tilt:
            double_reflected = tilt_rays(double_reflected, apply_tilt['tilt_deg'], apply_tilt['pa_deg'], inverse=True, 
                                         z_rotation = -telescope_pars['L1s'],
                                        xshift=apply_tilt['xshift'], yshift=apply_tilt['yshift'])
            # create_image_rays_of_type_(double_reflected, "Double reflected", '/Users/John/Desktop/tilted')
        
        rays_maps = dict(map(create_image_rays_of_type_, [failed_all_reflections, reflected_ray_only_iperbola, reflected_only_parabola, double_reflected],
                                                         ["Failed all reflections", "Reflect only iperbola", "Reflect only parabola", "Double reflected"]))
        
        if plot_tracing is not False:
            ax.axhline(-rays_maps["Double reflected"]['best_focal_plane'], color='green', lw=4)
        
        x_combined = np.array([])
        y_combined = np.array([])
        A_eff = 0
        for i in rays_maps.keys():
            # x_combined = np.append(x_combined, rays_maps[i]['x'])
            # y_combined = np.append(y_combined, rays_maps[i]['y'])
            if i in ["Double reflected"]:
                x_combined = np.append(x_combined, rays_maps[i]['x'])
                y_combined = np.append(y_combined, rays_maps[i]['y'])
                A_eff += rays_maps[i]['A_eff']

        if rugosity_key=='liscio': x_center, y_center = x_combined.mean(), y_combined.mean() #this way if surface is rugosa, we use the values from liscia surface
        hew = half_energy_diameter(x_combined, y_combined, f0s.mean(), x_center, y_center)

    return {'rays_maps':rays_maps, 'x':x_combined, 'y':y_combined, 'Aeff':A_eff, 'hew':hew, 'best_focal_plane':rays_maps["Double reflected"]['best_focal_plane'],
           'nrays_initial':nrays_initial}


def generator_f_wolterI_auto(R_initial, squared_size, f0, L, thickness, inner_mirror):
    """
    Generates shells for a Wolter I that fit inside a square starting from an initial radius, and where each parabola's outer radius is coincident with the 
    radius of the intersection plane of the next shell.

    Parameters
    ----------
    R_initial: float
        Radius of inner shell.
    squared size: float
        Diameter of the system.
    f0: float
        Focal length.
    L: float
        Length of each parabola and iperbola.
    thickness: float
        Tickness of mirrors.
    inner_mirror: bool
        If true, also the inner mirror for internal absorptions is computed.
    """
    radii_center = []
    radii_parabola = []
    
    Rc = R_initial
    Area = 0

    Rp = R_initial

    if Rp > (squared_size)/2:
        Rp = 0
    
    while Rp < (squared_size)/2:
    
        theta = np.arctan2(Rc, f0)/4
        beta = 3*theta

        Rp = np.sqrt(Rc**2 + np.tan(theta)*L*Rc)
    
        radii_center.append(Rc)
        radii_parabola.append(Rp) #raggi della parabola in alto, quindi lo spazio aperto dello specchio è compreso tra radii_center e radii_parabola
        
        Area += np.pi*(Rp**2 - Rc**2)
    
        Rc = Rp + thickness
    
    spessori = np.repeat(thickness, len(radii_center))
    
    L1s = np.repeat(L, len(radii_center))/2
    # raggi_center_inner = np.append((radii_center[1:])+spessori[1:], 0) + thickness #sembrerebbe essere sbgliato
    raggi_center_inner = np.append(0.00001, radii_center[:-1]+spessori[:-1])
    
    return {'radii_parabola':np.array(radii_parabola), 'radii_center':np.array(radii_center), 'radii_center_inner':raggi_center_inner, 
            'L1s':L1s, 'f0':f0, 'best_focal_plane':f0, 'inner_mirror':inner_mirror}   
