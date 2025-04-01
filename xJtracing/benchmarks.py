import numpy as np
import copy

def half_energy_diameter(x, y, f0=None, x_center=None, y_center=None):
    """
    The half energy diameter is a benchmark indicating the circular diameter in which
    half the counts are contained
    
    Parameters
    ----------
    x, y: array_like
        Coordinates of points on the detector.
    f0: float
        Focal length of optical system.
    x_center, y_center: float, optional
        Center of the circle from which to compute the half energy diameter.
    
    Notes
    -----
    Uses median around a center, which if not user-defined, is the mean.
    """
    if x_center is None: x_center = x.mean()
    if y_center is None: y_center = y.mean()
    diameters = 2*np.sqrt((x - x_center)**2 + (y - y_center)**2)
    if f0:
        return np.nanmedian(diameters)*180/np.pi*3600/f0
    else:
        return np.nanmedian(diameters)


def bins_around_mean(x, y, half_space = 10, increment = 0.01):
    """
    Parameters
    ----------
    x, y: np.ndarray
        Data points
    half_space: float
        The histogram is computed around x.mean +- half_space, same for y
    increment: flaot
        Pixel size of the histogram
    """
    return [np.arange(np.mean(x) - half_space, np.mean(x) + half_space, increment),
            np.arange(np.mean(y) - half_space, np.mean(y) + half_space, increment)]

def bins_from_data(x, y, increment=0.01):
    """
    Creates the optimal bins to bin all data.

    Parameters
    ----------
    x, y: np.ndarray
        Data x and y coordinates.
    increment: float
        Size of pixel of histogram.
    """
    return [np.arange(x.min(), x.max(), increment), np.arange(y.min(), y.max(), increment)]


def find_max_xy_from_hist(x, y, bins_function=bins_around_mean, **bins_kwargs):
    """
    Finds the center of an image using the brighest pixel of an histogram.
    
    Parameters
    ----------
    x, y: np.ndarray
        Data points
    bins_kwargs:
        Parameters of the bins_function function
    bins_function: function
        function computing the bins, taking in input x and y and bins_kwargs

    Return
    ------
    x_center, y_center: float
        Coordinates of center 
    """
    h, x_edgs, y_edgs = np.histogram2d(x, y, bins = bins_function(x, y, **bins_kwargs))
    maxx, maxy = np.where(h==h.max())
    x_center = x_edgs[maxx[0]]
    y_center = y_edgs[maxy[0]]
    return x_center, y_center



def mm_to_arcsec(radius):
    """
    Multiplicative parameter of mm to arcsec conversion give a certain radius [in mm].
    """
    return 180/np.pi*3600/(radius/2)

def filter_using_hist(data, centre, threshold = 'auto', side_size = 200):
    """
    Filters data by making an histogram and picking only points above a certain threshold.

    Parameters
    ----------
    data: np.ndarray
        Input data.
    centre: float
        Center around which to make the histogram.
    threshold: float or str
        Value above which data is selected. If not passed, it is assigne the value max/4.
    side_size: float
        Left and right padding of histogram.
    """
    hist, bin_edges = np.histogram(data, bins=np.linspace(centre - side_size, centre + side_size, side_size))
    bins = np.digitize(data, bin_edges)
    data[bins==0] = np.nan
    data[bins==bins.max()] = np.nan
    bins[bins==0] = 1
    bins[bins==bins.max()] = bins.max() - 1
    if threshold=='auto':
        threshold = hist.max()/4
    return hist[bins-1] > threshold

def filter_2d_using_hist(x, y, x_ctr1, y_ctr1):
    """
    Filter data in 2 dimensions using a threshold in a histogram.

    Parameters
    ----------
    x, y: np.ndarray
        Input data.
    x_ctr1, y_ctr1: float
        Center around which to make the histogram.
    """
    return np.logical_and.reduce(list(map(filter_using_hist, [x, y], [x_ctr1, y_ctr1])))


def hew_Lobster(rays_input, radius):
    """
    Half energy width using the cross thickness.

    Parameters
    ----------
    rays_input: dictionary containing 'x' and 'y' as np.ndarray
        Input x and y data on the detector.
    radius: float
        Radius of the lobster eye.
    """
    rays_maps = copy.deepcopy(rays_input)
    x, y = rays_maps['x']*mm_to_arcsec(radius), rays_maps['y']*mm_to_arcsec(radius)
    
    x_ctr1, y_ctr1 = find_max_xy_from_hist(x, y, bins_function=bins_from_data, increment=1)

    condition = filter_2d_using_hist(x, y, x_ctr1, y_ctr1)
    x_center_new = x[condition].mean()
    y_center_new = y[condition].mean()
    return half_energy_diameter(x[condition], y[condition], None, x_center_new, y_center_new), x_center_new, y_center_new


def cross_rays_fractions(rays_input, x_center, y_center):
    """
    Computes the fraction of rays that go in the cross, in the center, and in the background (both stray light and double reflected background).

    Parameters
    ----------
    rays_input: output of simulate_lobster_rays
        Input rays.
    x_center, y_center: float
        Coordinates of the center.
    """
    rays_maps = copy.deepcopy(rays_input)
    nevents = rays_maps['total']['x'].size
    
    try:
        rays1 = rays_maps[1]['x'].size
        rays1_x = rays_maps[1]['x']
        rays1_y = rays_maps[1]['y']
    except:
        rays1 = rays_maps['1up']['x'].size + rays_maps['1down']['x'].size
        rays1_x = np.append(rays_maps['1up']['x'], rays_maps['1down']['x'])
        rays1_y = np.append(rays_maps['1up']['y'], rays_maps['1down']['y'])
    
    cloud_mask =       (rays_maps[2]['x'] - x_center)**2 + (rays_maps[2]['y'] - y_center)**2 > 5**2
    center_only_mask = (rays_maps[2]['x'] - x_center)**2 + (rays_maps[2]['y'] - y_center)**2 < 5**2
    
    return {'fraction0':rays_maps[0]['x'].size/nevents,
            'fraction1':rays1/nevents,
            'fraction2':rays_maps[2]['x'].size/nevents, 
            'fraction_center': rays_maps[2]['x'][center_only_mask].size/nevents,
            'fraction_clouds': rays_maps[2]['x'][cloud_mask].size/nevents}

def all_hew_info_from_cross(rays_maps, radius):
    """
    Computes all the relevant information from the cross image and returns it in a dictionary.

    Input:
    rays_maps: output of simulate_lobster_rays
        Input rays.
    radius: float
        Radius of the lobster eye.
    """
    hew, x_center_new, y_center_new = hew_Lobster(rays_maps['total'], radius)
    
    x_center = x_center_new/mm_to_arcsec(radius)
    y_center = y_center_new/mm_to_arcsec(radius)
    dict_out = cross_rays_fractions(rays_maps, x_center, y_center)
    
    dict_out['hew'] = hew
    return dict_out


# LA PARTE LOBSTER QUI SOTTO `E DA RISISTEMARE: ORA E' DEPRECATAAAAAA
# ===================================================================


# def _find_center_with_uncertainty(h, x_edges, y_edges):
#     """
#     Start from 2d histogram of detector image h with its edges, 
#     do a weighted mean with uncertainty given by the Poisson statistics of each pixel.
#     """
    
#     x_centers = (x_edges[:-1] + x_edges[1:])/2
#     y_centers = (y_edges[:-1] + y_edges[1:])/2
    
#     def mediapesata_pesi_incerti(x, w, w_err):
#         x_mean = np.sum(w*x)/np.sum(w)
#         w_sum = np.sum(w)
#         wx_sum = np.sum(w*x)
#         sigma_x = np.sqrt(np.sum((((x*w_sum - wx_sum)/w_sum**2)*w_err)**2))
    
#         return x_mean, sigma_x
    
#     x_mean, sigma_x = mediapesata_pesi_incerti(x_centers, h.sum(axis=1), np.sqrt(h.sum(axis=1)))
#     y_mean, sigma_y = mediapesata_pesi_incerti(y_centers, h.sum(axis=0), np.sqrt(h.sum(axis=0)))
    
#     return x_mean, sigma_x, y_mean, sigma_y    


# def ray_density_experiment(x, y):
#     x.sort()
#     y.sort()
    
#     xdist = np.sum(x[1:] - x[:-1])
#     ydist = np.sum(y[1:] - y[:-1])

#     return x.size/xdist, y.size/ydist



# def hew_Lobster_DEPRECATED(rays_input, radius, hew_method, center_finder_coarsness = 0.1, debug_plot=False, padding = 70):
    
#     rays_maps = copy.deepcopy(rays_input)
    
#     assert hew_method in ['all cross', 'only center', 'use cross']
    
#     mm_to_arcsec = 180/np.pi*3600/(radius/2)
#     datax = rays_maps['total']['x']*mm_to_arcsec
#     datay = rays_maps['total']['y']*mm_to_arcsec
    
#     x_center = rays_maps['total']['x'].mean()
#     y_center = rays_maps['total']['y'].mean()
#     print('first centers found', x_center, y_center)

#     if rays_maps['total']['x'].size>0:

#         #trova centro
#         if debug_plot:
#             fig, ax = plt.subplots()
#             ax.hist2d(rays_maps['total']['x'], rays_maps['total']['y'], bins = [np.arange(x_center-padding, x_center+padding, center_finder_coarsness), 
#                                                   np.arange(y_center-padding, y_center+padding, center_finder_coarsness)])

#         h, x_edgs, y_edgs = np.histogram2d(rays_maps['total']['x'], rays_maps['total']['y'], 
#                                         bins = [np.arange(x_center-padding, x_center+padding, center_finder_coarsness), 
#                                                   np.arange(y_center-padding, y_center+padding, center_finder_coarsness)]
#                                  )
        
#         x_mean, sigma_x, y_mean, sigma_y = _find_center_with_uncertainty(h, x_edgs, y_edgs)
#         maxx, maxy = np.where(h==h.max())
#         print('max', maxx, maxy)
#         maxx, maxy = maxx[0], maxy[0]
#         x_center = (x_edgs[maxx] + x_edgs[maxx+1])/2
#         y_center = (y_edgs[maxy] + y_edgs[maxy+1])/2
#         print('second centers found', x_center, y_center)
#         hew_mask = (rays_maps[2]['x'] - x_center)**2 + (rays_maps[2]['y'] - y_center)**2 < 10**2
#         x_center_new = rays_maps[2]['x'][hew_mask].mean()*mm_to_arcsec
#         y_center_new = rays_maps[2]['y'][hew_mask].mean()*mm_to_arcsec
#         print('found new centers', x_center_new/mm_to_arcsec, y_center_new/mm_to_arcsec, x_center_new, y_center_new)

#         if debug_plot:
#             fig, ax = plt.subplots()
#             ax.hist2d(rays_maps[2]['x'][hew_mask], rays_maps[2]['y'][hew_mask], bins = [np.arange(x_center-padding, x_center+padding, center_finder_coarsness), 
#                                                   np.arange(y_center-padding, y_center+padding, center_finder_coarsness)])
#             fig, ax = plt.subplots(1,2)
#             for (i, data), centre in zip(enumerate([datax, datay]), [x_center_new, y_center_new]):
#                 ax[i].hist(data)
#                 ax[i].axvline(centre, color='red')
        
#         if hew_method == 'all cross':
#             hew = half_energy_diameter(datax, datay, None, datax.mean(), datay.mean())
#             x_center, y_center = np.mean(datax), np.mean(datay)
            
#         elif hew_method == 'only center':
#             hew = half_energy_diameter(rays_maps[2]['x'][hew_mask]*mm_to_arcsec, rays_maps[2]['y'][hew_mask]*mm_to_arcsec, None, x_center_new, y_center_new)
            
#         elif hew_method == 'use cross':
#             threshold = 'auto'
#             conditions = {}
#             for _xykey, data, centre in zip(['x', 'y'], [datax, datay], [x_center_new, y_center_new]):
#                 hist, bin_edges = np.histogram(data, bins=np.linspace(centre - 200, centre + 200, 200))
#                 bins = np.digitize(data, bin_edges)
#                 data[bins==0] = np.nan
#                 data[bins==bins.max()] = np.nan
#                 bins[bins==0] = 1
#                 bins[bins==bins.max()] = bins.max() - 1
#                 if threshold=='auto':
#                     threshold = hist.max()/4
#                 conditions[_xykey] = hist[bins-1] > threshold
#             condition = np.logical_and(conditions['x'], conditions['y'])
#             hew = half_energy_diameter(datax[condition], datay[condition], None, x_center_new, y_center_new)

#             if debug_plot:
#                 fig, ax = plt.subplots(1,2)
#                 for (i, data), centre in zip(enumerate([datax, datay]), [x_center_new, y_center_new]):
#                     ax[i].hist(data[condition])
#                     ax[i].axvline(centre, color='red')
        
#         nevents = rays_maps['total']['x'].size
#     else:
#         hew = None
#         x_mean, sigma_x, y_mean, sigma_y = 0,0,0,0
#         nevents = np.nan

#     def mm_to_arcsec(mm):
#         return np.arctan(mm/(radius/2))*180/np.pi*3600

#     cloud_mask =       (rays_maps[2]['x'] - x_center)**2 + (rays_maps[2]['y'] - y_center)**2 > 5**2
#     center_only_mask = (rays_maps[2]['x'] - x_center)**2 + (rays_maps[2]['y'] - y_center)**2 < 5**2
#     try:
#         rays1 = rays_maps[1]['x'].size
#         rays1_x = rays_maps[1]['x']
#         rays1_y = rays_maps[1]['y']
#     except:
#         rays1 = rays_maps['1up']['x'].size + rays_maps['1down']['x'].size
#         rays1_x = np.append(rays_maps['1up']['x'], rays_maps['1down']['x'])
#         rays1_y = np.append(rays_maps['1up']['y'], rays_maps['1down']['y'])
#     # try:
#     #     rays3 = rays_maps['3+']['x'].size
#     # except:
#     #     rays3 = 0

#     N_rays_useful = rays1 + rays_maps[2]['x'][center_only_mask].size
#     rays_useful_x = np.append(rays1_x, rays_maps[2]['x'][center_only_mask])
#     rays_useful_y = np.append(rays1_y, rays_maps[2]['y'][center_only_mask])
    
#     return {'hew': hew, 
#             # 'x_center': [np.round(x_mean, 4), '±', np.round(sigma_x, 4)],
#             # 'y_center': [np.round(y_mean, 4), '±', np.round(sigma_y, 4)],
#             'x_center_arcsec': [np.round(mm_to_arcsec(x_mean), 4), '±', np.round(mm_to_arcsec(sigma_x), 4)],
#             'y_center_arcsec': [np.round(mm_to_arcsec(y_mean), 4), '±', np.round(mm_to_arcsec(sigma_y), 4)],
#             'fraction0':rays_maps[0]['x'].size/nevents,
#             'fraction1':rays1/nevents,
#             'fraction2':rays_maps[2]['x'].size/nevents, 
#             'fraction_center': rays_maps[2]['x'][center_only_mask].size/nevents,
#             'fraction_clouds': rays_maps[2]['x'][cloud_mask].size/nevents,
#             'A_eff_without_clouds': rays_maps[0]['area_over_Nrays']*N_rays_useful, #+ rays3)
#             'ray_density_experiment_usefull': ray_density_experiment(rays_useful_x, rays_useful_y),
#             'ray_density_experiment_all': ray_density_experiment(rays_maps['total']['x'], rays_maps['total']['y']),
#             'ray_density_experiment_bkg': ray_density_experiment( np.append(rays_maps[0]['x'], rays_maps[2]['x'][cloud_mask]), 
#                                                                   np.append(rays_maps[0]['y'], rays_maps[2]['y'][cloud_mask]))
#            }

    
