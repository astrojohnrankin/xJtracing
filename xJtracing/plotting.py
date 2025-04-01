import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# TEMPORARY
import os, sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append("..")

from benchmarks import bins_around_mean


def plot_mirror(ax, mirror, inner=False, side='radial'):
    """
    Plots a mirror.

    Parameters
    ----------
    ax: axis
        Matplotlib axis.
    mirror: instance of a xJtracing.mirror data structure
        Mirror to be plotted. Limits are interpolated from z lims.
    inner: bool
        if True, mirror is plotted gray instead of gold.
    side: str
        Can be radial, x or y, 'x Lobster' or 'y Lobster'.
    """
    try:
        z = np.linspace(mirror.z_low, mirror.z_up, 10000)
        if side == 'radial':
            x = mirror.equation_x(0, z)
        elif side in ['x', 'x Lobster']:
            x = mirror.equation_x(z, 0)
        elif side in ['y', 'y Lobster']:
            x = mirror.equation_y(z, 0)
            
        if side in ['x Lobster', 'y Lobster']:
            n_mirrors = int(np.sqrt(mirror.z_low.shape)[0])
        if side in ['x Lobster']:
            x = x[:, :n_mirrors]
            z = z[:, :n_mirrors]
        if side in ['y Lobster']:
            x = x[:, ::n_mirrors]
            z = z[:, ::n_mirrors]
    except:
        raise
        if inner and mirror.z_low.size>1:
            x_low = optimize.fsolve(lambda x: np.append(0, mirror.equation(x, 0)[1:] - mirror.z_low[1:]), np.zeros(mirror.z_low.size))
            x_up = optimize.fsolve( lambda x: np.append(0, mirror.equation(x, 0)[1:] - mirror.z_up[1:] ), np.zeros(mirror.z_low.size))
        else:
            x_low = optimize.fsolve(lambda x: mirror.equation(x, 0) - mirror.z_low, np.zeros(mirror.z_low.size))
            x_up = optimize.fsolve(lambda x: mirror.equation(x, 0) - mirror.z_up, np.zeros(mirror.z_low.size))
    
        
        x = np.linspace(x_low, x_up, 10000)
        z = mirror.equation(x, 0)

    if inner:
        kwargs = {'color':'gray', 'lw':2}
    else:
        kwargs = {'color':'gold', 'lw':3}
    ax.plot(x, z, **kwargs)


def plot_rays(ax, rays, z_low, z_up, side='radial'):
    """
    Plots rays. Plotted only in a 1mm slice around x axis, otherwise plot would be confusing.

    Parameters
    ----------
    ax: axis
        Matplotlib axis.
    mirror: instance of a xJtracing.rays.rays_dataclass.
        Rays to be plotted
    z_low, z_up: float
        Range of existance of rays.

    """

    from rays import generate_ray_direction_equations
    
    if isinstance(z_low, (float, int)): z_low = np.tile(z_low, rays.survival.shape)
    if isinstance(z_up, (float, int)): z_up = np.tile(z_up, rays.survival.shape)
    
    n_rays = rays.survival.shape[0]
    n_configs = rays.survival.shape[1]

    if side in ['x Angel', 'y Angel']:
        n_mirrors = int(np.sqrt(n_configs))
    
    configs = range(n_configs)
    if side in ['x Angel']:
        configs = configs[:n_mirrors]
    if side in ['y Angel']:
        configs = configs[::n_mirrors]
    
    for config_i in configs:
        z = np.linspace(z_low[:,config_i], z_up[:,config_i], 3)
            
        x_func, y_func = generate_ray_direction_equations(rays.e[:,:,config_i], rays.x0[:,:,config_i])
        if side in ['y', 'y Angel']: #If True, the plot is in the yz space instead of xz.
            x_func, y_func = y_func, x_func

        slicing_list = [rays.delete_ray[:,config_i]==0, rays.survival[:,config_i]]
        if side in ['radial', 'x', 'y']:
            slicing_list.append(y_func(z_up[:,config_i]) > -1.5)
            slicing_list.append(y_func(z_up[:,config_i]) < 1.5)
        if side in ['radial']:
            slicing_list.append(x_func(z_up[:,config_i]) > 0)
        mask_slice = np.logical_and.reduce(slicing_list)
    
        ax.plot(x_func(z)[:,mask_slice], z[:,mask_slice], lw=0.5, color='C0')




def plot_rays_map(rays_maps, save_str=None, outdir='', rays_keys=range(3), padding=None):
    """
    Plots the map of rays, color-coded for the number of reflection.
    
    Parameters
    ----------
    rays_maps: dict
        Dictionary of rays reflected n times in order, with each item a dictionary with x and y points in mm on detector.
    save_str: str
        String of which to begin the filename.
    outdir: path
        Directory where to save plots.
    rays_keys: list
        Items of rays_maps to plot.
    padding: float
        If not None, the x and y limits of the plots are set according to [-padding, padding].
    """
    # try:
    fig, ax = plt.subplots()
    for j, i in enumerate(rays_keys):
        ax.plot(rays_maps[i]['x'], rays_maps[i]['y'], '.', color=f'C{j}', label=f'N reflections = {i}')
    if padding:
        ax.set_xlim(-padding, padding)
        ax.set_ylim(-padding, padding)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.legend()
    ax.set_aspect('equal')
    if save_str:
        # ax.set_title(save_str)
        fig.savefig(os.path.join(outdir, f'{save_str}_rays_map.pdf'))
    # except:
    #     raise
    #     print('no ray map possible')
        

def plot_image(x, y, save_str=None, outdir='None', padding = 70, pixel_size=0.05, norm='log', cmap='viridis'):
    """
    Plots the image of the telescope on the detector.
    
    Parameters
    ----------
    x, y: np.ndarray
        x and y points in mm on detector.
    save_str: str
        String of which to begin the filename.
    outdir: path
        Directory where to save plots.
    padding: float
        If not None, the x and y limits of the plots are set according to [-padding, padding].
    pixel_size: float
        Size of pixel.
    norm: str
        Normalization of image scale.
    cmap: str
        Colormap.
    """

    fig, ax = plt.subplots()
    h, x_edgs, y_edgs, _img = ax.hist2d(x, y, bins_around_mean(x, y, half_space = padding, increment = pixel_size), norm=norm, cmap=cmap)
    
    cbar = fig.colorbar(_img, ax=ax)
    cbar.ax.set_ylabel('counts/pixel')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_aspect('equal')
    if save_str:
        fig.savefig(os.path.join(outdir, f'{save_str}_image.png'), dpi=300)



def plot_image_OLD(x, y, save_str=None, outdir='None', padding = 70, pixel_size=0.05, norm='log', cmap='viridis'):
    """
    Plots the image of the telescope on the detector.
    
    Parameters
    ----------
    x, y: np.ndarray
        x and y points in mm on detector.
    save_str: str
        String of which to begin the filename.
    outdir: path
        Directory where to save plots.
    padding: float
        If not None, the x and y limits of the plots are set according to [-padding, padding].
    pixel_size: float
        Size of pixel.
    norm: str
        Normalization of image scale.
    cmap: str
        Colormap.
    """
    if not isinstance(padding, list):
        padding = [padding, padding]
    # try:
    x_center = x.mean()
    y_center = y.mean()
    fig, ax = plt.subplots()
    h, x_edgs, y_edgs, _img = ax.hist2d(x, y, bins = [np.arange(x_center-padding[0], x_center+padding[0], pixel_size), 
                                                  np.arange(y_center-padding[1], y_center+padding[1], pixel_size)]
                                                 ,norm=norm, cmap=cmap)
    cbar = fig.colorbar(_img, ax=ax)
    cbar.ax.set_ylabel('counts/pixel')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_aspect('equal')
    if save_str:
        fig.savefig(os.path.join(outdir, f'{save_str}_image.png'), dpi=300)
    # except:
    #     print('no image possible')





#bruttooooooooooo:

def plot_rays_flattened(ax, rays, z_low, z_up, side='radial'):
    """
    Plots rays. Plotted only in a 1mm slice around x axis, otherwise plot would be confusing.
    FLATTENED, WOULD BE NICE TO INCLUDE IT IN FUNCTION ABOVE
    Parameters
    ----------
    ax: axis
        Matplotlib axis.
    mirror: instance of a xJtracing.rays.rays_dataclass.
        Rays to be plotted
    z_low, z_up: float
        Range of existance of rays.

    """

    from rays import generate_ray_direction_equations
    
    if isinstance(z_low, (float, int)): z_low = np.tile(z_low, rays.survival.shape)
    if isinstance(z_up, (float, int)): z_up = np.tile(z_up, rays.survival.shape)
    
    n_rays = rays.survival.shape[0]

    z = np.linspace(z_low[:], z_up[:], 3)
        
    x_func, y_func = generate_ray_direction_equations(rays.e[:,:], rays.x0[:,:])
    if side in ['y', 'y Angel']: #If True, the plot is in the yz space instead of xz.
        x_func, y_func = y_func, x_func

    slicing_list = [rays.delete_ray[:]==0, rays.survival[:]]
    if side in ['radial', 'x', 'y']:
        slicing_list.append(y_func(z_up[:]) > -1.5)
        slicing_list.append(y_func(z_up[:]) < 1.5)
    if side in ['radial']:
        slicing_list.append(x_func(z_up[:]) > 0)
    mask_slice = np.logical_and.reduce(slicing_list)

    ax.plot(x_func(z)[:,mask_slice], z[:,mask_slice], lw=0.5, color='C0')