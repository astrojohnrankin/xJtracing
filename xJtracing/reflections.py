import numpy as np

def find_tangent_plane_and_normal(mirror_surface_z, x_intersect, y_intersect, epsilon=1e-8):
    """
    Plane tangent to mirror in a given point.
    
    Parameters
    ----------
    mirror_surface_z: function(x, y: array_like) -> array_like
        Function describing the curved surface over which we find tangend and normal.
    x_intersect, y_intersect: array_like
        Intersection points where to start the plane and normal.
    epsilon: float
        step over which to do the derivative.
        
    Returns
    -------
    tan_plane_z: function(x, y: array_like) -> array_like
        Function describing the tangent plane
    n: array of 3 array_like
        Versor normal to the plane

    Notes
    -----
    The partial derivatives over x and y are computed numerically. The tangent plane it then given by
        
    .. math:: z = \\frac{\\partial f}{\\partial x} (x - x_0) + \\frac{\\partial f}{\\partial y} (y - y_0) + z_0

    The versor of the normal surface is then given by
    
    .. math::
    
    	v = [\\frac{\\partial f}{\\partial x}, \\frac{\\partial f}{\\partial y}, -1]
    """
    df_dx = (mirror_surface_z(x_intersect+epsilon, y_intersect) - mirror_surface_z(x_intersect-epsilon, y_intersect))/(2*epsilon)
    df_dy = (mirror_surface_z(x_intersect, y_intersect+epsilon) - mirror_surface_z(x_intersect, y_intersect-epsilon))/(2*epsilon)
    def tan_plane_z(x,y):
        return df_dx*(x-x_intersect) + df_dy*(y-y_intersect) + mirror_surface_z(x_intersect, y_intersect)
    
    v = np.array([df_dx, df_dy, -np.ones(df_dx.shape)])
    n = v/np.sqrt(np.sum(v**2, axis=0))
    
    return tan_plane_z, n


def find_normal_to_flat_plane(alpha, theta, x_intersect, y_intersect, z_intersect):
    """
    Line normal to a plane (we use this function when dealing with a plane, while we use 
    the find_tangent_plane_and_normal function when dealing with curved surfaces).
    
    Parameters
    ----------
    alpha, theta: array_like
        Angles [rad] defining the flat plane, see definitions in 
        mirror.generate_flat_surface.
        
    Returns
    -------
    n: array of 3 array_like
        Verson normal to the plane.
    """
    n = np.array([np.cos(alpha)*np.cos(theta)*np.ones(x_intersect.shape),
                  np.cos(alpha)*np.sin(theta)*np.ones(x_intersect.shape),
                  -np.sin(alpha)*np.ones(x_intersect.shape)])
    return n


def angle_two_vectors(vec1, vec2):
    """
    Angle between two vectors.
    
    Parameters
    ----------
    vec1, vec2: array of 3 array_like
        Vectors of which to find angle in between
    
    Returns
    -------
    float
        Angle [rad] between the two vectors.
    """
    return np.pi/2 - np.arccos(np.sum(vec1 * vec2, axis=0))
    
    
def reflect_ray(e, n_normal, x_intersect, y_intersect, z_intersect):
    """
    Reflected ray.
    
    Parameters
    ----------
    e: array of 3 array_like
        Versor of input ray(s).
    n_normal: array of 3 array_like
        Versor of normal to surface on which reflection occurs.
    x_intersect, y_intersect, z_intersect: array_like
        Intersection point of ray(s) and plane
    
    Returns
    -------
    array of 3 array_like
        Versor of reflected ray(s)

    Notes
    -----
    This function is derived as follows. The incident versor :math:``\\vec{k}_{0}`
    is written as the sum of it's ortogonal components
    
    .. math::
        \\vec{k}_{0}=\\vec{k}_{0\\parallel}+\\vec{k}_{0\\perp}
    
    :math:`\\vec{k}_{0\\perp}` can be rewritten from :math:`\\vec{k}_{0}` using the
    normal versor :math:`\\vec{n}`
    
    .. math::
        \\vec{k}_{0\\perp}=\\left|\\vec{k}_{0\\perp}\\right|\\vec{n}=\\left(\\vec{k}_{0}\\cdot\\vec{n}\\right)\\vec{n}
    
    so that 
    
    .. math::
        \\vec{k}_{0}=\\vec{k}_{0\\parallel}+\\left(\\vec{k}_{0}\\cdot\\vec{n}\\right)\\vec{n}
    
    for the reflected ray, the expression is similar but with a different
    sign for the ortogonal component
    
    .. math::
        \\vec{k}_{1}=\\vec{k}_{0\\parallel}-\\left(\\vec{k}_{0}\\cdot\\vec{n}\\right)\\vec{n}
    
    from which
    
    .. math::
        \\vec{k}_{1}=\\vec{k}_{0}-\\left(\\vec{k}_{0}\\cdot\\vec{n}\\right)\\vec{n}

    """
    e_out = e - 2*np.sum(e*n_normal, axis=0)*n_normal
    return e_out


def rotate_normal_for_rugosity(rugosity, n, e):
    """
    Scattering over a rugose surface causes a change in the direction of the normal, 
    which we simulate here.
    
    Parameters
    ----------
    rugosity: float
        The half energy width [rad] of the Lorentzian spread due to rugosity
    n: tuple of 3 array_like
        Normal to the plane
    e: tuple of 3 array_like
        Versor of the ray(s) hitting the surface over which they are reflected
    
    Returns
    -------
    tuple of 3 array_like
        Normal to the plane rotated for scattering
        
    Notes
    -----
    We already have the normal direction :math:`n`, we derive the parallel :math:`t` (see the notes in the reflect_ray function).
    The scattering is described by a Lorentzian function.
    """
    
    t = e - np.sum(e*n, axis=0)*n
    rugosity_a = np.random.standard_cauchy(e.shape)*rugosity/2 #assumes rugosity=HEW
    alpha = np.cos(rugosity_a/2)
    beta = np.sin(rugosity_a/2)
    n = alpha*n + beta*t
    n = n/np.sqrt(np.sum(n**2, axis=0)) 
    return n


def rotate_ray(e, theta):
    """
    Rotate ray around y.
    
    Parameters
    ----------
    e: tuple of 3 array_like
        Initial array versor.
    theta: float
        Rotation angle [rad]
    """

    e_vec = np.array([
                np.cos(theta)*e[0] + np.sin(theta)*e[2],
                e[1],
                -np.sin(theta)*e[0] + np.cos(theta)*e[2]
            ])                  
    return e_vec
