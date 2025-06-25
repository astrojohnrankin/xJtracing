import numpy as np
import matplotlib.pyplot as plt
import copy
import os, sys
import pandas as pd
from functools import reduce
from xJtracing.rays import reflect_ray_starting_from_mirror, mark_rays_for_non_deletion, delete_rays_marked_for_deletion, tilt_rays, save_rays_to_file
from xJtracing.generators import create_circular_corona_of_rays_for_WI
from xJtracing.intersection import create_image#, find_best_focal_plane
from xJtracing.tracing_utils import assert_array_1d_single
from xJtracing.mirror import parabola_dataclass, iperbola_dataclass
from xJtracing.benchmarks import half_energy_diameter
from xJtracing.plotting import plot_mirror, plot_rays
from xJtracing.data.paths import path_erosita


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


def select_params_for_shell(telescope_pars, shell_i):
    """
    Takes as input the multi-shell dictionary telescope_pars and returns the parameters only
    of the shell with index shell_i,
    """
    def select_par_i_(key_):
        if key_ == 'apply_tilt':
            return 'apply_tilt', select_params_for_shell(telescope_pars['apply_tilt'], shell_i)
        if isinstance(telescope_pars[key_], (np.ndarray, list)):
            return key_, telescope_pars[key_][shell_i:shell_i+1]
        else:
            return key_, telescope_pars[key_]
    return dict(map(select_par_i_, telescope_pars.keys()))


def run_single_WI_shell(off_axis_angle_deg, pa_deg, telescope_pars, rays_function, energy, rays_in_mm2,
                       material_nk_files_list, d_list, rugosity_hew, plot_tracing):
    """
    Simulation of a single Wolter-I shell. For the parameters see xJtracing.Wolter_i.simulate_a_WolterI.
    """
    assert_array_1d_single(telescope_pars['f0'], telescope_pars['radii_center'], telescope_pars['L1s'], 
                         telescope_pars['theta'], telescope_pars['beta'])
    parabola = parabola_dataclass(R0=telescope_pars['radii_center'], theta=telescope_pars['theta'], z_low = 0, z_up = telescope_pars['L1s'])
    iperbola = iperbola_dataclass(R0=telescope_pars['radii_center'], beta=telescope_pars['beta'], theta=telescope_pars['theta'], f0=telescope_pars['f0'], z_low = -telescope_pars['L1s'], z_up = 0)
    if telescope_pars['inner_mirror']:
        parabola_inner = parabola_dataclass(R0=telescope_pars['radii_center_inner'], theta=telescope_pars['theta'], z_low = 0, z_up = telescope_pars['L1s'])
        iperbola_inner = iperbola_dataclass(R0=telescope_pars['radii_center_inner'], beta=telescope_pars['beta'], theta=telescope_pars['theta'], f0=telescope_pars['f0'], z_low = -telescope_pars['L1s'], z_up = 0)
    else:
        parabola_inner, iperbola_inner = None, None
    
    rays = rays_function(angle_off_axis=off_axis_angle_deg*np.pi/180, telescope_pars=telescope_pars, 
                                              rays_in_mm2=rays_in_mm2, energy=energy, pa_angle=pa_deg*np.pi/180)
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
        
    if 'apply_tilt' in telescope_pars:
        rays = tilt_rays(rays, telescope_pars['apply_tilt']['tilt_deg'], telescope_pars['apply_tilt']['pa_deg'])

    failed_all_reflections, reflected_ray_only_iperbola, reflected_only_parabola, double_reflected = simulate_rays_Wolter_I(rays=rays, 
                                                                        material_nk_files_list=material_nk_files_list, d_list=d_list, parabola=parabola, iperbola=iperbola, 
                                                                        inner_mirror=telescope_pars['inner_mirror'], parabola_inner=parabola_inner, iperbola_inner=iperbola_inner, rugosity=rugosity_hew, ax=ax)

    if plot_tracing is not False:
        # for rays_to_plot in [reflected_ray_only_iperbola]:
        for rays_to_plot in [failed_all_reflections, reflected_ray_only_iperbola, reflected_only_parabola, double_reflected]:
            plot_rays(ax, rays_to_plot, -telescope_pars['f0'], rays_to_plot.x0[2])
        # ax.axhline(-telescope_pars['f0'], color='green', lw=4)


    if 'apply_tilt' in telescope_pars:
        double_reflected = tilt_rays(double_reflected, telescope_pars['apply_tilt']['tilt_deg'], 
                                     telescope_pars['apply_tilt']['pa_deg'], inverse=True, 
                                     z_rotation = -telescope_pars['L1s'],
                                     xshift=telescope_pars['apply_tilt']['xshift'], yshift=telescope_pars['apply_tilt']['yshift'])


    dict_rays = {"Failed all reflections":failed_all_reflections, 
            "Reflect only iperbola":reflected_ray_only_iperbola, 
            "Reflect only parabola":reflected_only_parabola, 
            "Double reflected":double_reflected,
                "nrays_initial":nrays_initial}
    return dict_rays
        
        
def simulate_a_WolterI(off_axis_angle_deg, pa_deg, telescope_pars, rays_function=create_circular_corona_of_rays_for_WI, energy=1, rays_in_mm2=50,
                       material_nk_files_list = ['data/nk/Au.nk'], d_list = [], rugosity_hew = False,
                       optimize_focal_plane=False, plot_tracing=False, which_rays_to_output = ["Double reflected"]):
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
        Finds the focal plane that minimizes the hew for the given off_axis_angle_deg.
    plot_tracing: bool
        If True, ray tracing is plotted.
    which_rays_to_output: list of str
        Which rays are used in the output, needs to be a list containing one or more of:
        ["Failed all reflections", "Reflect only iperbola", "Reflect only parabola", "Double reflected"]

    Notes
    -----
    The angles of the parabola and iperbola :math:`\\alpha` and :math:`\\theta` are not given in input, inside telescope_pars, they are instead derived in this function to achieve the
    maximum reflection efficiency:
    
    .. math::
    	\\theta = \\arctan{\\frac{R}{f_0}}/4
    	\\beta = 3\\theta
    """
    
    telescope_pars['theta'] = np.arctan2(telescope_pars['radii_center'], telescope_pars['f0'])/4
    telescope_pars['beta'] = 3*telescope_pars['theta']
    n_shells = telescope_pars['radii_parabola'].size
    parameters_divided_by_shell = list(map(lambda n_shells: select_params_for_shell(telescope_pars, n_shells),
                                           range(n_shells)))
    if plot_tracing is True: 
        fig, plot_tracing = plt.subplots()
        
    def run_single_WI_shell_(telescope_pars_i, rugosity_hew_):
        return run_single_WI_shell(off_axis_angle_deg, pa_deg, telescope_pars_i, rays_function, energy, rays_in_mm2,
                       material_nk_files_list, d_list, rugosity_hew_, plot_tracing)
        
    run_single_WI_shell_liscio = lambda telescope_pars_i: run_single_WI_shell_(telescope_pars_i, False)
    all_shells_output_liscio = list(map(run_single_WI_shell_liscio, parameters_divided_by_shell))
    rays_maps_list = [all_shells_output_liscio]
    if rugosity_hew:
        run_single_WI_shell_rugoso = lambda telescope_pars_i: run_single_WI_shell_(telescope_pars_i, rugosity_hew)
        all_shells_output_rugoso = list(map(run_single_WI_shell_rugoso, parameters_divided_by_shell))
        rays_maps_list.append(all_shells_output_rugoso)

    def last_steps_of_simulation(optimize_focal_plane, f0):
        if optimize_focal_plane:
            f0_delta = 100
            precision = 0.1
            def hew_at_this_f_(f_):
                return last_steps_of_simulation(False, f_)['hew']
            fs = np.arange(f0 - f0_delta, f0 + f0_delta, precision)
            hews = np.array(list(map(hew_at_this_f_, fs)))
            try:
                best_focal_plane = fs[hews==hews.min()][0]
            except:
                print('failed focal_plane adjustment')
                best_focal_plane = f0
        else:
            best_focal_plane = f0
            if plot_tracing is not False:
                plot_tracing.axhline(-telescope_pars['f0'], color='green', lw=4)

        
        for i, rays_maps in enumerate(rays_maps_list):
            
            def combine_shells_outputs(shell_i, rays_label="Double reflected"):
                rays_for_angles = delete_rays_marked_for_deletion(rays_maps[shell_i][rays_label])
                incidence_angles = rays_for_angles.incidence_angles #questi raggi non eliminanon quelli assorbiti dall'assorbimento degli angoli
                
                reflected_rays_passed = mark_rays_for_non_deletion(rays_maps[shell_i][rays_label], rays_maps[shell_i][rays_label].survival==True)
                rays_final = delete_rays_marked_for_deletion(reflected_rays_passed)
                x_det, y_det, z_det = create_image(rays_final.e, rays_final.x0, best_focal_plane)
                Aeff_i = rays_final.area_over_Nrays*rays_final.e[0].size #dovrebbe essere solo di questi specifici raggi riflessi N volte
                nrays_initial_i = rays_maps[shell_i]['nrays_initial']
                return x_det, y_det, incidence_angles, Aeff_i, rays_maps[shell_i]["nrays_initial"]
    
            # xyAn = list(map(combine_shells_outputs, range(n_shells)))
            xyAn_function = lambda label, which_rays_to_output_: (label, list(map(combine_shells_outputs, list(range(n_shells)) * len(which_rays_to_output_), 
                                                                                        [item for item in which_rays_to_output_ for _ in range(n_shells)])))
            xyAn = dict(map(xyAn_function, ['selected', 'total', '0', 'iperbola', 'parabola', '2'], 
                                            [which_rays_to_output, 
                                            ["Failed all reflections", "Reflect only iperbola", "Reflect only parabola", "Double reflected"], 
                                            ["Failed all reflections"], ["Reflect only iperbola"], ["Reflect only parabola"], ["Double reflected"]]))
            x_, y_, incidence_angles, A_, n_ = zip(*xyAn['selected'])
            x_combined = reduce(np.append, x_)
            y_combined = reduce(np.append, y_)
            A_eff = np.array(A_).sum()
            nrays_initial = np.array(n_).sum()
    
            if i==0: x_center, y_center = x_combined.mean(), y_combined.mean() #this way if surface is rugosa, we use the values from liscia surface
            
            x__, y__, incidence_angles__, A__, n__ = zip(*xyAn['total'])
            x_combined_total = reduce(np.append, x__)
            def get_fraction(label):
                x__, y__, incidence_angles__, A__, n__ = zip(*xyAn[label])
                x_combined_i = reduce(np.append, x__)
                return x_combined_i.size/x_combined_total.size
            hew_dict = {'fraction0':get_fraction('0'), 
                        'fraction1':get_fraction('iperbola') + get_fraction('parabola'), 
                        'fraction_center':get_fraction('2')}
            
            
    
        hew = half_energy_diameter(x_combined, y_combined, best_focal_plane, x_center, y_center)
    
        return {'rays_maps':rays_maps, 'x':x_combined, 'y':y_combined, 'incidence_angles':incidence_angles, 'Aeff':A_eff, 'hew':hew, 'best_focal_plane':best_focal_plane,
               'nrays_initial':nrays_initial, 'hew_dict':hew_dict}
            
    return last_steps_of_simulation(optimize_focal_plane, telescope_pars['best_focal_plane'])


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



def erosita_shells(every_n=1):
    """
    Returns the geometrical parameters of eROSITA shells, also skipping shells if needed (this is why this function is inside tilting, where skipping shells is useful).
    """
    tab_eRosita = pd.read_csv(path_erosita, sep='\s+')
    spessori = tab_eRosita['thk(mm)'].values
    weights = tab_eRosita['m(kg)'].values
    radii_c = 0.5*np.array([348.483, 338.522, 328.799, 319.406, 310.232, 301.378, 292.733, 284.394, 276.251, 268.401, 260.738, 253.256, 246.049, 239.015, 232.149, 225.542, 219.094, 212.800, 206.752, 200.850, 195.086, 189.461, 184.067, 178.801, 173.661, 168.744, 163.940, 159.253, 154.675, 150.208, 145.944, 141.783, 137.719, 133.754, 129.881, 126.202, 122.606, 119.099, 115.674, 112.331, 109.166, 106.077, 103.060, 100.117, 97.242, 94.435, 91.694, 89.119, 86.607, 84.152, 81.751, 79.350, 
                          76.949, 74.549])
    radii_max =  0.5*np.array([356.528, 346.338, 336.391, 326.782, 317.401, 308.342, 299.499, 290.966, 282.637, 274.607, 266.766, 259.112, 251.741, 244.545, 237.518, 230.760, 224.164, 217.724, 211.538, 205.498, 199.602, 193.846, 188.328, 182.940, 177.681, 172.649, 167.735, 162.940, 158.256, 153.685, 149.323, 145.066, 140.909, 136.851, 132.890, 129.123, 125.447, 121.858, 118.353, 114.932, 111.695, 108.534, 105.449, 102.436, 99.495, 96.622, 93.819, 91.184, 88.613, 86.100, 83.646, 81.189, 
                          78.732, 76.275])
    
    assert isinstance(every_n, int)
    shell_selection_mask = np.arange(spessori.size) % every_n == 0
    radii_c, radii_max, spessori = shells_selector(radii_c, radii_max, spessori, shell_selection_mask)


    return radii_c, radii_max, spessori, weights[shell_selection_mask]



def shells_selector(radii_c, radii_max, spessori, shell_selection_mask, weights=None):
    """
    Selects one shell from the arrays of geometrical parameters.
    """
    radii_c = radii_c[shell_selection_mask]
    radii_max = radii_max[shell_selection_mask]
    spessori = spessori[shell_selection_mask]

    return radii_c, radii_max, spessori


def bilancia_kg(radii_center_mm, L_mm, spessore_mm, f_mm, rho_g_cm3=8.9):
    """
    Computes weigth of a Wolter-I optical system.
    """

    if f_mm is not None:
        theta = np.arctan2(radii_center_mm, f_mm)/4
        beta = 3*theta
        L_up_mm = L_mm/np.cos(theta)
        L_low_mm = L_mm/np.cos(beta)
        L_tot_cm = L_up_mm/10 + L_low_mm/10
    else:
        L_tot_cm = 2*L_mm/10
    
    single_shell_volume = 2*np.pi*radii_center_mm/10 * L_tot_cm * spessore_mm/10
    
    return np.sum(single_shell_volume)/1000*rho_g_cm3