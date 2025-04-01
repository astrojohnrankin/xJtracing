import numpy as np

#TEMPORARY
import os, sys
# sys.path.append(os.path.dirname(__file__))
from xJtracing.rays import generate_ray_direction_equations
from xJtracing.benchmarks import half_energy_diameter

def _intersection_analitical_general(A, B, C, x_0, e, bounds_z, bounds_x='undefined', bounds_y='undefined', use_x_y_bounds=False):
    """
    Support function for analitical intersections. 

    Notes
    -----
    A ray is defined as 
    
    .. math::
    	\\vec x = \\vec x_0 + \\vec e * t
    
    This function solves the equation
    
    .. math::
		A*t**2 + B*t + C = 0
    It also takes into account if the found intersection point is inside the boundaries.
    """
    A = np.where(np.abs(A)<1e-10, 0, A) #When tilting and de-tilting, 0 values become something like 1e-18, which is !=0 and causes the division by A to go to very high values, when instead it should divide by A when A is supposed to be 0
    
    e1, e2, e3 = e
    x0, y0, z0 = x_0
    
    _x_intersect = lambda t: x0 + e1*t
    _y_intersect = lambda t: y0 + e2*t
    _z_intersect = lambda t: z0 + e3*t
    
    exists_intersection = np.tile(True, A.shape) 
            
    t = np.where(A==0, -C/B, np.empty(A.shape))
    if use_x_y_bounds: 
        bounds_list = [bounds_z, bounds_x, bounds_y]
    else: bounds_list = [bounds_z]
    for _intersect, _bounds in zip([_z_intersect, _x_intersect, _y_intersect], bounds_list):
        exists_intersection = np.where(np.logical_and(A==0, np.logical_or(_intersect(t) < _bounds[0], _intersect(t) > _bounds[1])), False, exists_intersection) #no intersection (CASE A==0)

    delta = B**2 - 4*A*C
    exists_intersection = np.where(delta < 0, False, exists_intersection) #delta negative while doing intersection analytically
    delta_avoids_warning = np.where(delta >= 0, delta, np.nan)
    A_avoids_warning = np.where(A==0, np.nan, A)
    tplus =  (-B + np.sqrt(delta_avoids_warning))/(2*A_avoids_warning)
    tminus = (-B - np.sqrt(delta_avoids_warning))/(2*A_avoids_warning)
    
    tplus_valid_z =  np.logical_and(_z_intersect(tplus)  > bounds_z[0], _z_intersect(tplus)  < bounds_z[1])
    tminus_valid_z = np.logical_and(_z_intersect(tminus) > bounds_z[0], _z_intersect(tminus) < bounds_z[1])
    if use_x_y_bounds:
        tplus_valid_x =  np.logical_and(_x_intersect(tplus)  > bounds_x[0], _x_intersect(tplus)  < bounds_x[1])
        tminus_valid_x = np.logical_and(_x_intersect(tminus) > bounds_x[0], _x_intersect(tminus) < bounds_x[1])
        tplus_valid_y =  np.logical_and(_y_intersect(tplus)  > bounds_y[0], _y_intersect(tplus)  < bounds_y[1])
        tminus_valid_y = np.logical_and(_y_intersect(tminus) > bounds_y[0], _y_intersect(tminus) < bounds_y[1])
    if use_x_y_bounds:
        tplus_valid = np.logical_and.reduce([tplus_valid_z, tplus_valid_x, tplus_valid_y])
        tminus_valid = np.logical_and.reduce([tminus_valid_z, tminus_valid_x, tminus_valid_y])
    else:
        tplus_valid = tplus_valid_z
        tminus_valid = tminus_valid_z
    exists_intersection = np.where(np.logical_and(tplus_valid, tminus_valid), False, exists_intersection) #two intersections inside bounds
    t = np.where(np.logical_and(A!=0,  tplus_valid), tplus, t)
    t = np.where(np.logical_and(A!=0,  tminus_valid), tminus, t)
    exists_intersection = np.where(np.logical_and(A!=0, np.logical_and(~tplus_valid, ~tminus_valid)), False, exists_intersection) #no intersection inside bounds
    
    x_intersect = _x_intersect(t)
    y_intersect = _y_intersect(t)
    z_intersect = _z_intersect(t)
    return (x_intersect, y_intersect, z_intersect), exists_intersection


def generate_parabola_intersection_function(R0, theta, z_low, z_up):
    """
    Generates the intersection function of a Wolter parabola with a ray.
    
    Parameters
    ----------
    R0, theta: array_like
        Parameters of the parabola, see its definition.
    z_low, z_up: array_like
        Upper and lower limits in z coordinates of where to look for the intersection.
        
    Returns
    -------
    function(
        x_0, e: array_like
            Parameters of the ray, see its definition in rays.generate_ray_direction_equations.
        ) -> 
            intersection_point: tuple of 3 array_like
                Coordinates of intersection point
            exists_intersection: array_like
                Mask indicating where the intersection exists (in case of dealing with many rays)
    
    Notes
    -----
    We start from the parabola equation
    
    .. math::
		\\frac{x^{2}+y^{2}}{R_{0}^{2}}=1+2\\tan\\left(\\theta\\right)\\frac{z}{R_{0}}
    
    solving we obtain the parameters of a quadratic equation
    
    .. math::
		\\frac{x^{2}+y^{2}}{R_{0}}=R_{0}+2\\tan\\left(\\theta\\right)z
    
		z=-\\frac{R_{0}}{2\\tan\\left(\\theta\\right)}+\\frac{x^{2}+y^{2}}{2\\tan\\left(\\theta\\right)R_{0}}
    	
    	a=\\frac{1}{2\\tan\\left(\\theta\\right)R_{0}}\\qquad c=-\\frac{R_{0}}{2\\tan\\left(\\theta\\right)}
    
    	z=a\\left(x^{2}+y^{2}\\right)+c
    
    
    substituting inside the parametrized rays we get
    
    .. math::
    	z_{0}+e_{3}t=a\\left(\\left(x_{0}+e_{1}t\\right)^{2}+\\left(y_{0}+e_{2}t\\right)^{2}\\right)+c
    
    
    this function solves this last equation for the :math:`t` at which intersection
    occurs.
    
    If input is a vector multiple rays can be simulated. A simulation can be performed
    with R0, theta, z_low and z_up as 1d arrays (all shells or configurations) and x_0 and 
    e as 2d arrays (first coordinate indicating which ray and second which shell or
    configuration).
    """
    def intersection_function(x_0, e):
        e1, e2, e3 = e
        x0, y0, z0 = x_0
        aa = 1/(2*np.tan(theta)*R0) #1d [n_configurations]
        cc = -R0/(2*np.tan(theta))
        A = aa*(e1**2 + e2**2) #2d [n_rays_max, n_configurations]
        B = aa*2*x0*e1 + aa*2*y0*e2 - e3
        C = cc - z0 + aa*(x0**2 + y0**2)
        return _intersection_analitical_general(A, B, C, x_0, e, bounds_z=[z_low, z_up])
    return intersection_function


def generate_parabola_KB_intersection_function(f, L, R0, axis, z_low, z_up, bounds_x, bounds_y):
    """
    Generates the intersection function of a Kirkpatrick-Baetz parabola with a ray.
    
    Parameters
    ----------
    f, L, R0, axis: array_like
        Parameters of the KB parabola, see its definition.
    z_low, z_up: array_like
        Upper and lower limits in z coordinates of where to look for the intersection.
        
    Returns
    -------
    function(
        x_0, e: array_like
            Parameters of the ray, see its definition in rays.generate_ray_direction_equations.
        ) -> 
            intersection_point: tuple of 3 array_like
                Coordinates of intersection point
            exists_intersection: array_like
                Mask indicating where the intersection exists (in case of dealing with many rays)
    
    Notes
    -----
    We start from the parabola equation

    
    .. math::
    	z=\\frac{r^{2}}{8f_{0}}+f-f_{0}
    
    solving we obtain the parameters of a quadratic equation
    
    .. math::
    	0=-z_{0}-e_{3}t+\\frac{x_{0}^{2}+y_{0}^{2}+e_{1}^{2}t^{2}+e_{2}^{2}t^{2}+2x_{0}e_{1}t+2y_{0}e_{2}t}{8f_{0}}+f-f_{0}
    
    	A=\\frac{e_{1}^{2}+e_{2}^{2}}{8f_{0}}\\qquad B=-e_{3}+\\frac{2x_{0}e_{1}+2y_{0}e_{2}}{8f_{0}}\\qquad C=-z_{0}+\\frac{x_{0}^{2}+y_{0}^{2}}{8f_{0}}+f-f_{0}
    
    
    substituting inside the parametrized rays we get the equation for
    which this function solves for the :math:`t` at which intersection occurs.
    
    If input is a vector multiple rays can be simulated. A simulation can be performed
    with R0, theta, z_low and z_up as 1d arrays (all shells or configurations) and x_0 and 
    e as 2d arrays (first coordinate indicating which ray and second which shell or
    configuration).
    """
    def intersection_function(x_0, e):
        e1, e2, e3 = e
        x0, y0, z0 = x_0
        f0 = (np.sqrt((f-L)**2 + R0**2/2) - (L - f))/2
        if axis=='x':
            A = (e1**2)/8/f0
            B = -e3 + (2*x0*e1)/8/f0
            C = -z0 + (x0**2)/8/f0 + f - f0
        elif axis=='y':
            A = (e2**2)/8/f0
            B = -e3 + (2*y0*e2)/8/f0
            C = -z0 + (y0**2)/8/f0 + f - f0
        elif axix=='both':
            A = (e1**2 + e2**2)/8/f0
            B = -e3 + (2*x0*e1 + 2*y0*e2)/8/f0
            C = -z0 + (x0**2 + y0**2)/8/f0 + f - f0
        return _intersection_analitical_general(A, B, C, x_0, e, bounds_z=[z_low, z_up], bounds_x=bounds_x, bounds_y=bounds_y, 
                                               use_x_y_bounds=True)
    return intersection_function
    
    
def generate_iperbola_intersection_function(R0, beta, theta, f0, z_low, z_up):
    """
    Generates the intersection function of a Wolter iperbola with a ray.
    
    Parameters
    ----------
    R0, beta, theta: array_like
        Parameters of the iperbola, see its definition.
    z_low, z_up: array_like
        Upper and lower limits in z coordinates of where to look for the intersection.
        
    Returns
    -------
    function(
        x_0, e: array_like
            Parameters of the ray, see its definition in rays.generate_ray_direction_equations.
        ) -> 
            intersection_point: tuple of 3 array_like
                Coordinates of intersection point
            exists_intersection: array_like
                Mask indicating where the intersection exists (in case of dealing with many rays)
    
    Notes
    -----
    We start from the iperbola equation

    
    .. math::
    	\\frac{x^{2}+y^{2}}{R_{0}^{2}}=1-2\\tan\\left(\\beta\\right)\\frac{z}{R_{0}}+\\frac{2R_{0}\\tan\\left(\\beta\\right)}{f_{0}+R_{0}/\\tan\\left(2\\theta\\right)}\\left(\\frac{z}{R_{0}}\\right)^{2}
    
    solving we obtain the parameters of a quadratic equation
    
    .. math::
    	x^{2}+y^{2}=R_{0}^{2}-R_{0}2\\tan\\left(\\beta\\right)z+\\frac{2R_{0}\\tan\\left(\\beta\\right)}{f_{0}+R_{0}/\\tan\\left(2\\theta\\right)}z^{2}
    
    	x^{2}+y^{2}=c+bz+az^{2}

    	c=R_{0}^{2}\\qquad b=-R_{0}2\\tan\\left(\\beta\\right)\\qquad a=\\frac{2R_{0}\\tan\\left(\\beta\\right)}{f_{0}+R_{0}/\\tan\\left(2\\theta\\right)}
    
    substituting inside the parametrized rays we get
    
    .. math::
    	\\left(x_{0}+e_{1}t\\right)^{2}+\\left(y_{0}+e_{2}t\\right)^{2}=c+b\\left(z_{0}+e_{3}t\\right)+a\\left(z_{0}+e_{3}t\\right)^{2}
    
    	x_{0}^{2}+e_{1}^{2}t^{2}+2x_{0}e_{1}t+y_{0}^{2}+e_{2}^{2}t^{2}+2y_{0}e_{2}t=c+bz_{0}+be_{3}t+az_{0}^{2}+ae_{3}^{2}t^{2}+a2z_{0}e_{3}t
    	
    	0=-x_{0}^{2}-e_{1}^{2}t^{2}-2x_{0}e_{1}t-y_{0}^{2}-e_{2}^{2}t^{2}-2y_{0}e_{2}t+c+bz_{0}+be_{3}t+az_{0}^{2}+ae_{3}^{2}t^{2}+a2z_{0}e_{3}t
    
    	0=t^{2}\\left(-e_{1}^{2}-e_{2}^{2}+ae_{3}^{2}\\right)+t\\left(-2x_{0}e_{1}-2y_{0}e_{2}+be_{3}+a2z_{0}e_{3}\\right)+\\left(-x_{0}^{2}-y_{0}^{2}+c+bz_{0}+az_{0}^{2}\\right)
    
    this function solves this last equation for the :math:`t` at which intersection
    occurs.

    If input is a vector multiple rays can be simulated. A simulation can be performed
    with surface parameters as 1d arrays (all shells or configurations) and x_0 and 
    e as 2d arrays (first coordinate indicating which ray and second which shell or
    configuration).
    """
    def intersection_function(x_0, e):
        e1, e2, e3 = e
        x0, y0, z0 = x_0
        a = (2*R0*np.tan(beta))/(f0 + R0/np.tan(2*theta))
        b = R0*2*np.tan(beta)
        c = R0**2
        A = -e1**2 - e2**2 + a*e3**2
        B = -2*x0*e1 - 2*y0*e2 + b*e3 + a*2*z0*e3
        C = -x0**2 - y0**2 + c + b*z0 + a*z0**2
        return _intersection_analitical_general(A, B, C, x_0, e, bounds_z=[z_low, z_up])
    return intersection_function
    

def generate_flat_surface_intersection_function(alpha, theta, mirror_x0, mirror_y0, mirror_z0, z_low, z_up, bounds_x, bounds_y):
    """
    Generates the intersection function of a flat surface with a ray.
    
    Parameters
    ----------
    alpha, theta, mirror_x0, mirror_y0, mirror_z0: array_like
        Parameters of the flat surface, see its definition.
    z_low, z_up: array_like
        Upper and lower limits in z coordinates of where to look for the intersection.
    bounds_y, bounds_z: functions returning tuple of 2 array_like
        Upper and lower limits in x coordinates of where to look for the intersection.
        
    Returns
    -------
    function(
        x_0, e: array_like
            Parameters of the ray, see its definition in rays.generate_ray_direction_equations.
        ) -> 
            intersection_point: tuple of 3 array_like
                Coordinates of intersection point
            exists_intersection: array_like
                Mask indicating where the intersection exists (in case of dealing with many rays)
    
    Notes
    -----
    We start from the flat surface equations
    
    
    .. math::
    	x=\\tan\\left(\\alpha\\right)\\left(z-z_{0}\\right)\\cos\\left(\\theta\\right)+\\sin\\left(\\theta\\right)v+x_{0}
    
    	y=\\tan\\left(\\alpha\\right)\\left(z-z_{0}\\right)\\sin\\left(\\theta\\right)+\\cos\\left(\\theta\\right)v+y_{0}
    
    
    substituting inside the parametrized rays we get 
    
    .. math::
    	x_{v0}+\\frac{e_{1}}{e_{3}}\\left(z-z_{0}\\right)=\\tan\\left(\\alpha\\right)\\left(z-z_{0}\\right)\\cos\\left(\\theta\\right)+\\sin\\left(\\theta\\right)v+x_{0}
    
    	y_{v0}+\\frac{e_{2}}{e_{3}}\\left(z-z_{0}\\right)=\\tan\\left(\\alpha\\right)\\left(z-z_{0}\\right)\\sin\\left(\\theta\\right)+\\cos\\left(\\theta\\right)v+y_{0}
    
    which then becomes
    
    .. math::
    	-x_{v0}-\\frac{e_{1}}{e_{3}}z+\\frac{e_{1}}{e_{3}}z_{0}+\\tan\\left(\\alpha\\right)\\cos\\left(\\theta\\right)z-\\tan\\left(\\alpha\\right)\\cos\\left(\\theta\\right)z_{0}+\\sin\\left(\\theta\\right)v+x_{0}=0
	
    	-y_{v0}-\\frac{e_{2}}{e_{3}}z+\\frac{e_{2}}{e_{3}}z_{0}+\\tan\\left(\\alpha\\right)\\sin\\left(\\theta\\right)z-\\tan\\left(\\alpha\\right)\\sin\\left(\\theta\\right)z_{0}+\\cos\\left(\\theta\\right)v+y_{0}=0
    
    	A_{x}+B_{x}z=\\sin\\left(\\theta\\right)\\nu
    
	    A_{y}+B_{y}z=\\cos\\left(\\theta\\right)\\nu
    
    	z=\\frac{\\frac{A_{y}}{\\cos\\left(\\theta\\right)}-\\frac{A_{x}}{\\sin\\left(\\theta\\right)}}{\\frac{B_{x}}{\\sin\\left(\\theta\\right)}-\\frac{B_{y}}{\\cos\\left(\\theta\\right)}}
    
    	A_{x}=-\\left(-x_{v0}+\\frac{e_{1}}{e_{3}}z_{0}-\\tan\\left(\\alpha\\right)\\cos\\left(\\theta\\right)z_{0}+x_{0}\\right)\\qquad B_{x}=-\\left(-\\frac{e_{1}}{e_{3}}+\\tan\\left(\\alpha\\right)\\cos\\left(\\theta\\right)\\right)
    
    	A_{y}=-\\left(-y_{v0}+\\frac{e_{2}}{e_{3}}z_{0}-\\tan\\left(\\alpha\\right)\\sin\\left(\\theta\\right)z_{0}+y_{0}\\right)\\qquad B_{x}=-\\left(-\\frac{e_{2}}{e_{3}}+\\tan\\left(\\alpha\\right)\\sin\\left(\\theta\\right)\\right)
    
    This function implements all of this (but note that the notation is
    slightly different, :math:`t` instead of :math:`v` and :math:`z_{0v}` or :math:`z_{0t}`instead
    of :math::math:`z_{0}` from flat surface.
    
    If input is a vector multiple rays can be simulated. A simulation can be performed
    with surface parameters as 1d arrays (all shells or configurations) and x_0 and 
    e as 2d arrays (first coordinate indicating which ray and second which shell or
    configuration).
    """
    from mirror import generate_flat_surface
    
    def intersection_function(x_0, e):
        e1, e2, e3 = e
        x0, y0, z0 = x_0
        
        mirror_surface_x, mirror_surface_y = generate_flat_surface(alpha, theta, mirror_x0, mirror_y0, mirror_z0)
        
        Ax = -(-x0 + e1/e3*z0 - np.tan(alpha)*np.cos(theta)*mirror_z0 + mirror_x0)
        Ay = -(-y0 + e2/e3*z0 - np.tan(alpha)*np.sin(theta)*mirror_z0 + mirror_y0)
        
        Bx = -(-e1/e3 + np.tan(alpha)*np.cos(theta))
        By = -(-e2/e3 + np.tan(alpha)*np.sin(theta))

        theta_avoids_warning_0 = np.where(theta==0, np.nan, theta)
        theta_avoids_warning_pi2 = np.where(theta==np.pi/2, np.nan, theta)
        z_intersect = (Ay/np.cos(theta_avoids_warning_pi2) - Ax/np.sin(theta_avoids_warning_0))/(Bx/np.sin(theta_avoids_warning_0) - By/np.cos(theta_avoids_warning_pi2))
        t_intersect = Ax + Bx*z_intersect

        Bx_avoids_warning = np.where(Bx==0, np.nan, Bx)
        z_intersect = np.where(theta==0, -Ax/Bx_avoids_warning, z_intersect)
        t_intersect = np.where(theta==0, Ay + By*z_intersect, t_intersect)

        By_avoids_warning = np.where(By==0, np.nan, By)
        z_intersect = np.where(theta==np.pi/2, -Ay/By_avoids_warning, z_intersect)
        t_intersect = np.where(theta==np.pi/2, Ax + Bx*z_intersect, t_intersect)

        x_intersect = mirror_surface_x(z_intersect, t_intersect)
        y_intersect = mirror_surface_y(z_intersect, t_intersect)
        
        exists_intersection = np.tile(True, Ax.shape)
        exists_intersection[z_intersect < z_low] = False
        exists_intersection[z_intersect > z_up] = False
        
        bounds_x_low, bounds_x_up = bounds_x(z_intersect)
        exists_intersection[x_intersect < bounds_x_low] = False
        exists_intersection[x_intersect > bounds_x_up] = False
        
        bounds_y_low, bounds_y_up = bounds_y(z_intersect)
        exists_intersection[y_intersect < bounds_y_low] = False
        exists_intersection[y_intersect > bounds_y_up] = False
        
        return (x_intersect, y_intersect, z_intersect), exists_intersection
    
    return intersection_function
    

def create_image(e, x_0, f0):
    """
    Creates image of rays that intersect flat detector at f0.
    
    Parameters
    ----------
    e, x_0: tuple of 3 array_like
        Rays parameters, see rays.generate_ray_direction_equations.
    f0: float
        Focal distance of detector.
    
    Returns
    -------
    x_det, y_det, z_det: array_like
        Coordinates of points on the detector.
    """
    rays_shape = e[0].shape
    # detector_surface_z = lambda x, y: np.tile(-f0, x.shape)
    f0s = np.tile(-f0, [rays_shape[0], 1])

    ray_direction_x, ray_direction_y = generate_ray_direction_equations(e, x_0)
    x_det, y_det, z_det = ray_direction_x(-f0), ray_direction_y(-f0), f0s
        
    return x_det, y_det, z_det


def find_best_focal_plane(reflected_rays_passed, f0, f0_delta, precision=0.1, plot=False):
    """
    Finds the focal plane that minizes the hew for the given off_axis_angle_deg.

    Parameters
    ----------
    reflected_rays_passed: instance of rays.rays_dataclass
        rays that go to the detector.
    f0: float
        Initial guess of focal distance.
    f0_delta: float
        The best focal distance is searched for in the interval f0 +- f0_delta.
    precision: float
        Precision of computed focal values.
    plot: bool
        Debugging plot.
    """

    def hew_at_this_f_(focal_plane):
        x_det, y_det, z_det = create_image(reflected_rays_passed.e, reflected_rays_passed.x0, focal_plane)
        x_center, y_center = x_det.mean(), y_det.mean()
        return half_energy_diameter(x_det, y_det, focal_plane, x_center, y_center)

    fs = np.arange(f0 - f0_delta, f0 + f0_delta, precision)
    hews = np.array(list(map(hew_at_this_f_, fs)))

    try:
        new_f = fs[hews==hews.min()][0]
    except:
        print('failed focal_plane adjustment')
        new_f = f0
        
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(fs, hews)
        ax.axvline(f0, color='red')
        ax.axvline(new_f, color='green')
    
    return new_f
    