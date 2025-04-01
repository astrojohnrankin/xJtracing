import numpy as np

#TEMPORARY
import os, sys
# sys.path.append(os.path.dirname(__file__))
from xJtracing.tracing_utils import assert_array_1d

def generate_Wolter_parabola(R0, theta):
    """
    Generate the equation for a polynomial parabola of Wolter I telescope.
    
    Parameters
    ----------
    R0: array_like
        Radius [mm] at the intersection plane (intersection between parabola and iperbola).
    theta: array_like
        Angle [rad] between the mirror and the optical axis at the intersection plane.
        
    Returns
    -------
    mirror_surface_z: function
        Parameters:
            x, y: array_like
                input coordinates
        Returns:
            z coordinate
    
    Notes
    -----
    A parabola for a Wolter-I telescope can be described, in radial coordinates, as
    
    .. math::
		\\frac{r_{}^{2}}{r_{0}^{2}}=1 + 2\\tan\\left(\\theta\\right)\\left(\\frac{z}{r_{0}}\\right)

    from which 
    
    .. math::
		z=-\\frac{r_{0}^{2}-r^{2}}{2\\tan\\left(\\theta\\right)r_{0}}
    
    If input is a vector multi-shell optics can be simulated.
    """
    def mirror_surface_x(y, z):
        return np.sqrt(z*2*np.tan(theta)*R0 + R0**2 - y**2)
    def mirror_surface_y(z, x):
        return np.sqrt(z*2*np.tan(theta)*R0 + R0**2 - x**2)
    def mirror_surface_z(x, y):
        return -(R0**2 - (x**2+y**2))/(2*np.tan(theta)*R0)
    return mirror_surface_x, mirror_surface_y, mirror_surface_z


def generate_KB_parabola(f, L, R0, axis):
    """
    Generate the equation for a polynomial parabola of Kirkpatrick-Baetz telescope.
    
    Parameters
    ----------
    f: array_like
        Focal length [mm] of the telescope.
    L: array_like
        Distance [mm] on z to R0
    R0: array_like
        Radius [mm] at L
    axis: str
        Can have values 'x', 'y' or 'both' according if the parabola is only over 
        an axis, or if it is a rotation parabola.
    Returns
    -------
    mirror_surface_z: function
        Parameters:
            x, y: array_like
                input coordinates
        Returns:
            z coordinate
    
    Notes
    -----
    This parabola is defind starting from the focus, length and radius of mirrors.
    
    .. math::
    	z = \\frac{x^2 + y^2}{8f_0} + f - f_0
    
    where 
    
    .. math::
		f0 = \\frac 1 2 (\\sqrt{(f-L)^2 + R_0^2/2} - (L - f))
    
    If input is a vector multi-shell optics can be simulated.
    """
    assert axis in ['x', 'y', 'both']
    f0 = (np.sqrt((f-L)**2 + R0**2/2) - (L - f))/2
    
    def mirror_surface_z(x, y):
        if axis=='x': 
            return (x**2)/8/f0 + f - f0
        elif axis=='y':
            return (y**2)/8/f0 + f - f0
        elif axis=='both':
            return (x**2+y**2)/8/f0 + f - f0

    def mirror_surface_x(z, y):
        if axis=='x': 
            return np.sqrt((z + f0 - f)*8*f0)
        elif axis=='y':
            return 0*(y+z)
        elif axis=='both':
            return np.sqrt((z + f0 - f)*8*f0 - y**2)

    def mirror_surface_y(z, x):
        if axis=='x': 
            return 0*(z+x)
        elif axis=='y':
            return np.sqrt((z + f0 - f)*8*f0)
        elif axis=='both':
            return np.sqrt((z + f0 - f)*8*f0 - x**2)
        
    return mirror_surface_x, mirror_surface_y, mirror_surface_z


def generate_Wolter_iperbola(R0, beta, theta, f0, sign=-1):
    """
    Generate the equation for a polynomial iperbola of Wolter I telescope.

    Parameters
    ----------
    R0: array_like
        Radius [mm] at the intersection plane (intersection between parabola and iperbola).
    beta: array_like
        Angle [rad] between the mirror and the optical axis at the intersection plane.
    theta: array_like
        Angle [rad] between the parabola (not iperbola) and the optical axis at the 
        intersection plane.
    f0: array_like
        Focal length [mm] of the telescope.
    sign: int
        To get the correct side of the iperbola, defaults to -1.
        
    Returns
    -------
    mirror_surface_z: function
        Parameters:
            x, y: array_like
                input coordinates
        Returns:
            z coordinate
    
    Notes
    -----
    An iperbola for a Wolter-I telescope can be described, in radial coordinates, as
    
    .. math::
    	\\frac{r^{2}}{r_{0}^{2}}=1-2\\tan\\left(\\beta\\right)\\left(\\frac{z}{r_{0}}\\right)+\\frac{2r_{0}\\tan\\left(\\beta\\right)}{f+r_{0}\\cot\\left(2\\theta\\right)}\\left(\\frac{z}{r_{0}}\\right)^{2}
    
    which can be rewritten as  
    
    .. math::
    	\\frac{2r_{0}\\tan\\left(\\beta\\right)}{f+r_{0}\\cot\\left(2\\theta\\right)}\\left(\\frac{z}{r_{0}}\\right)^{2}-2\\tan\\left(\\beta\\right)\\left(\\frac{z}{r_{0}}\\right)+\\left(1-\\frac{r_{2}^{2}}{r_{0}^{2}}\\right)=0
    This is II order equation that can be easily solved for z, as this function does.
    
    If input is a vector multi-shell optics can be simulated.
    """
    a = 2*R0*np.tan(beta)/(f0 + R0/np.tan(2*theta))/R0**2
    b = -2*np.tan(beta)/R0
    def mirror_surface_x(y, z):
        return np.sqrt(R0**2 * (1 - b*z + a*z**2) - y**2)
    def mirror_surface_y(z, x):
        return np.sqrt(R0**2 * (1 - b*z + a*z**2) - x**2)
    def mirror_surface_z(x, y):
        R1_2 = x**2 + y**2
        c = 1-R1_2/R0**2
        return -(-b + sign*np.sqrt(b**2 - 4*a*c))/(2*a)
    return mirror_surface_x, mirror_surface_y, mirror_surface_z


def generate_flat_surface(alpha, theta, x0, y0, z0=0):
    """
    Generates the (parametric) equations for a flat surface.
    
    Parameters
    ----------
    alpha, theta: array_like
        Coordinates [rad] of the inclination of the surface
    x0, y0, z0: array_like
        Coordinates [mm] of a point through which the surface passes
        
    Returns
    -------
    flat_surface_x, flat_surface_y: functions
        Parameters:
            z: array_like
                input z coordinate
            t: array_like
                parametric parameters
        Returns:
            x or y coordinate
    
    Notes
    -----
    We start by defining the parameters

    .. math::
    	\\alpha=\\text{inclination of plane}
    
    	\\theta=\\text{orientation of plane}
    
    
    The equations can be parametrized using two parameters :math:`u` and :math:`t`
    
    .. math::
		x=\\sin\\left(\\alpha\\right)u+\\sin\\left(\\theta\\right)t+x_{0}
	
    	y=\\sin\\left(\\alpha\\right)u+\\cos\\left(\\theta\\right)t+y_{0}

	    z=\\cos\\left(\\alpha\\right)u+z_{0}

    
    By susbtitution the parameter :math:`u` disappears
    
    .. math::
		x=\\tan\\left(\\alpha\\right)\\left(z-z_{0}\\right)\\cos\\left(\\theta\\right)+\\sin\\left(\\theta\\right)t+x_{0}
	
		y=\\tan\\left(\\alpha\\right)\\left(z-z_{0}\\right)\\sin\\left(\\theta\\right)+\\cos\\left(\\theta\\right)t+y_{0}
    
    If input is a vector multi-shell or cellular optic can be simulated.
    """
    def flat_surface_x(z, t):
        return np.tan(alpha)*(z - z0)*np.cos(theta) + np.sin(theta)*t + x0
    def flat_surface_y(z, t):
        return np.tan(alpha)*(z - z0)*np.sin(theta) + np.cos(theta)*t + y0
    return flat_surface_x, flat_surface_y
    
    
# Data structures to organize the above definitions:
# ==================================================

class parabola_dataclass:
    """
    Data structure containing the parabola properties.
    
    Parameters
    ----------
    R0, theta: array_like 1d [n configurations]
        Parabola parameters, see their definition in xJtracing.mirror
    z_low, z_up: array_like 1d [n configurations]
        Boundaries where parabola exists
    """
    
    def __init__(self, R0, theta, z_low, z_up):
        from intersection import generate_parabola_intersection_function
        assert_array_1d(R0, theta, z_low, z_up)
        self.equation_x, self.equation_y, self.equation = generate_Wolter_parabola(R0=R0, theta=theta)
        self.intersection_function = generate_parabola_intersection_function(R0=R0, 
                                                    theta=theta, z_low=z_low, z_up=z_up)
        self.z_low = z_low
        self.z_up = z_up


class KB_parabola_dataclass:
    """
    Data structure containing the parabola suited for Kirkpatrick Baetz properties.

    Parameters
    ----------
    f, L, R0, axis: array_like 1d [n configurations]
        Parabola parameters, see their definition in xJtracing.mirror
    z_low, z_up: array_like 1d [n configurations]
        Boundaries where parabola exists
    bounds_x, bounds_y: 2 item lists of functions of z returning a list of min and max boundaries
        Functions defining the boundaries.
    """
    def __init__(self, f, L, R0, axis, z_low, z_up, bounds_x, bounds_y):
        # assert_array_1d(f, L, R0, z_low, z_up)
        from intersection import generate_parabola_KB_intersection_function
        self.equation_x, self.equation_y, self.equation = generate_KB_parabola(f=f, L=L, R0=R0, axis=axis)
        self.intersection_function = generate_parabola_KB_intersection_function(f=f, L=L, R0=R0, axis=axis, z_low=z_low, z_up=z_up, 
                                                                                bounds_x=bounds_x, bounds_y=bounds_y)
        self.z_low = z_low
        self.z_up = z_up


class iperbola_dataclass:
    """
    Data structure containing the iperbola properties.
    
    Parameters
    ----------
    R0, beta, theta, f0: array_like 1d [n configurations]
        Iperbola parameters, see their definition in xJtracing.mirror
    z_low, z_up: array_like 1d [n configurations]
        Boundaries where iperbola exists
    """
    def __init__(self, R0, beta, theta, f0, z_low, z_up):
        from intersection import generate_iperbola_intersection_function
        assert_array_1d(R0, beta, theta, f0, z_low, z_up)
        self.equation_x, self.equation_y, self.equation = generate_Wolter_iperbola(R0=R0, beta=beta, theta=theta, f0=f0)
        self.intersection_function = generate_iperbola_intersection_function(R0=R0, 
                                   beta=beta, theta=theta, f0=f0, z_low=z_low, z_up=z_up)
        self.z_low = z_low
        self.z_up = z_up
                                   
                                   
class flat_mirror_dataclass:
    """
    Data structure containing the flat mirror.

    Parameters
    ----------
    alpha, theta, x0, y0, z0: array like 1d [n configurations]
        Flat surface parameters, see their definition in xJtracing.mirror
    z_low, z_up: array_like 1d [n configurations]
        Boundaries where parabola exists
    bounds_x, bounds_y: 2 item lists of functions of z returning a list of min and max boundaries
        Functions defining the boundaries.
    """
    def __init__(self, alpha, theta, x0, y0, z0, z_low, z_up, bounds_x, bounds_y):
        from intersection import generate_flat_surface_intersection_function
        self.alpha = alpha
        self.theta = theta
        self.bounds_x = bounds_x
        self.bounds_y = bounds_y
        self.z_low = z_low
        self.z_up = z_up
        self.x0, self.y0 = x0, y0
        self.equation_x, self.equation_y = generate_flat_surface(alpha=alpha, theta=theta, x0=x0, y0=y0, z0=z0)
        self.intersection_function = generate_flat_surface_intersection_function(alpha=alpha, 
                                                    theta=theta, mirror_x0=x0, mirror_y0=y0, 
                                                    mirror_z0=z0, z_low=z_low, z_up=z_up, 
                                                    bounds_x=bounds_x, bounds_y=bounds_y)
        