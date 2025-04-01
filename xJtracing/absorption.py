import numpy as np
import pandas as pd
from dataclasses import dataclass
from astropy import units
from scipy.interpolate import interp1d
import itertools


def generate_refractive_index_function(material_nk_file):
    """
    Generates the complex refractive index function.

    Parameters
    ----------
    material_nk_file: path
        Table file containing refraction index real and imaginary components as a function 
        of energy. If it is None a refractive index of 1 is returned.
        
    Returns
    -------
    function
        Function taking as input the energy and returning the refraction index, 
        with parameters
            E: array_like
                Energy [keV]
        and returning
            complex
                Refraction index
        
    Notes
    -----
    The refractive index as a function of energy is written as
    
    .. math::
        n(E) = d(E) + k(E)\\cdot j
        
    where the real part :math:`d` indicates the bending due to refraction (and is given by the 
    ratio of the speed of light and phase velocity in the medium), while the complex part
    :math:`k` is the (mainly photoelectric) absorption coefficient, proportional to the density 
    of the material. For X-rays, low Z materials have little absorption but reflect only
    lower energies (but so good for coatings), high Z materials have a little less 
    reflectivity but keep it up to higher energies.
        
    """
    if material_nk_file is None:
        def n_function(E):
            return 1 + 0*E
    else:
        material_table = pd.read_csv(material_nk_file, sep='\s+', skiprows=8, header=None, 
                                     names=['l', 'n', 'k'])
        lambda_nk, n_nk, k_nk = material_table['l'].values*units.angstrom, material_table['n'].values, material_table['k'].values
        keV_nk = lambda_nk.to(units.keV, equivalencies=units.spectral()).value  # Convert Angstrom to keV
        d_function = interp1d(keV_nk, n_nk, kind='linear')
        k_function = interp1d(keV_nk, k_nk, kind='linear')
        def n_function(E):
            return d_function(E) + k_function(E)*1j
    return n_function



def generate_Fresnel_coefficients(material0_nk_file, material1_nk_file):
    """
    Generates the Fresnel coefficients
    
    Parameters
    ----------
    material_nk_file: path
        Table file containing refraction index real and imaginary components as a function 
        of energy. If it is None a refractive index of 1, such as for air, is assumed.
    
    Returns
    -------
    function
         Fresnel coefficients as a function of inclination angle and energy, with
         parameters:
            alpha_deg: array_like
                Inclination angle [rad]
            energy: array_like
                Energy [keV]
        which returns the dataclass
            g0, g1: Parameters inside the Fresnel equations
            Rs, Rp: Respectively ortogonal and parallel reflectances
            alpha_crit: total reflection angle [rad] 
    
    Notes
    -----
    Fresnel coefficients describe the reflection and transmission of light between two
    different materials.
    The total reflection angle is where :math:`\\cos(\\alpha_{crit}) = n`
    """
    
    @dataclass(frozen=True)
    class Fresnel_coefficients_data:
       g0: float
       g1: float
       Rs: float
       Rp: float
       alpha_crit: float
    
    n0_function = generate_refractive_index_function(material0_nk_file)
    n1_function = generate_refractive_index_function(material1_nk_file)
    
    def Fresnel_coefficients(alpha, E):
        g0 = np.sqrt(n0_function(E)**2 - np.cos(alpha)**2)
        g1 = np.sqrt(n1_function(E)**2 - np.cos(alpha)**2)
        Rs = (g0 - g1)/(g0 + g1)
        Rp = (-g0/n0_function(E)**2 + g1/n1_function(E)**2)/(g0/n0_function(E)**2 + g1/n1_function(E)**2)
        alpha_crit = np.arccos(n1_function(E)/n0_function(E))
        return Fresnel_coefficients_data(g0, g1, Rs, Rp, alpha_crit)
    return Fresnel_coefficients
    
    
def reflection_R_many_layers(inc_angle, energy, d_list, material_nk_files_list):
    """
    Calculates the reflectivity between many layers considering interference (?).
    
    Parameters
    ----------
    inc_angle: array_like
        Inclination angle of incident radiation on top layer
    energy: array_like
        Energy [keV] of incident radiation
    d_list: list of float
        Thicknesses of layers (the top layer should not be in this list because it is
        considered to be infinite)
    material_nk_files_list: list of paths
        Table files containing refraction index real and imaginary components as a function 
        of energy. If it is None a refractive index of 1, such as for air, is assumed.
        
    Returns
    -------
    Rs, Rp: array_like
        Reflectivity for the two polarizations
    
    Notes
    -----
    Interface between two materials, for example
    r0
    r1
    r2
    r3
    The function starts with the 3 lowest layers (3-2-1) and computes reflectivity 
    between 1 and 2 (interference with 3), this value is then used for the interface 
    0 - 1 (interference with 2, 3), and so on with more layers if there were.
    
    """
    def recursive_R(layer_i, pr12='uncalculated', pp12='uncalculated', r12_uncalculated=True):
        
        material0_nk_file = material_nk_files_list[layer_i + 0]
        material1_nk_file = material_nk_files_list[layer_i + 1]
        material2_nk_file = material_nk_files_list[layer_i + 2]
        
        d1 = d_list[layer_i]
        
        if r12_uncalculated:
            reflect_specific12 = generate_Fresnel_coefficients(material1_nk_file, material2_nk_file)(inc_angle, energy)
            pr12, pp12 = reflect_specific12.Rs, reflect_specific12.Rp 
         
        reflect_specific01 = generate_Fresnel_coefficients(material0_nk_file, material1_nk_file)(inc_angle, energy)
        
        pr01, pp01 = reflect_specific01.Rs, reflect_specific01.Rp
        energy_keV = energy*units.keV
        lambda0 = energy_keV.to(units.angstrom, equivalencies=units.spectral()).value
        delta_phi = 2*np.pi/lambda0*2*reflect_specific01.g1*d1

        def R_function(r01, r12):
            return ( r01 + r12*np.exp(delta_phi*1j))/(1 + r01*r12*np.exp(delta_phi*1j)) 
        Rs = R_function(pr01, pr12)
        Rp = R_function(pp01, pp12)
        
        @dataclass(frozen=True)
        class Fresnel_coefficients_data:
           Rs: float
           Rp: float
        
        if layer_i==0:
            return Fresnel_coefficients_data(Rs=Rs, Rp=Rp)
        else:
            return recursive_R(layer_i-1, pr12=Rs, pp12=Rp, r12_uncalculated=False)

    return recursive_R(len(material_nk_files_list)-3)
    

def multilayers_from_receipt(high_Z_path, low_Z_path, bottom_path, N, a, b, c, G):
    """
    Creates the multilayers thicknesses.
    
    Parameters
    ----------
    high_Z_path, low_Z_path, bottom_path: paths
        Paths to the alternating and bottom materials nk files
    N: int
        number of layers
    a, b, c, G: float
        Multilayer receipts parameters, see notes
        
    Returns
    -------
    d_list: list
        List of ticknesses [Angstrom]
    material_nk_files_list:
        List of alternating materials with emptiness on top.
        
    Notes
    -----
    The total tickness of each bi-layer is given by
        d(k) = a/(b+k)^c
    and the ticknesses of the two alternating layers are then
        d_high_Z = G*d(k)
        d_low_Z = (1 - G)*d(k)
    k=1 indicates the topmost layer.
    """
    def tickensses(k):
        d = a/(b + k + 1)**c
        d_highZ = G*d
        d_lowZ = (1 - G)*d
        return d_lowZ, d_highZ
        
    d_list = list(itertools.chain.from_iterable(tickensses(i) for i in range(N)))
    
    material_nk_files_list = [
        None,
        *itertools.chain.from_iterable((low_Z_path, high_Z_path) for _ in range(N)),
        bottom_path
    ]
    
    return d_list, material_nk_files_list
    