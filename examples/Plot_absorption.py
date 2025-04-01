# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# %matplotlib widget

# +
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# -----------------------------------------
#TEMPORARY until I create packages
import sys
sys.path.append('..')
sys.path.append('.')
# -----------------------------------------

from xJtracing.absorption import generate_refractive_index_function, generate_Fresnel_coefficients, reflection_R_many_layers, multilayers_from_receipt

# +
datapath = '../xJtracing/data/nk'

try:
    datapath = os.path.basename(os.path.dirname(__file__))+'/../xJtracing/data/nk'
except:
    datapath = '../xJtracing/data/nk'
# -

# # Refractive index

n = generate_refractive_index_function(os.path.join(datapath, 'Au.nk'))

fig, ax = plt.subplots()
E = np.linspace(0.1, 10, 1000)
ax.plot(E, n(E))
ax.set_xlabel('Energy [keV]')
ax.set_ylabel('n')

# # Fresnel coefficients

Fresnel_equations = generate_Fresnel_coefficients(None, os.path.join(datapath, 'Au.nk'))

Fresnel_equations(89*np.pi/180, np.array([0.1,1.1, 4.1]))

E = np.linspace(0.1, 10, 1000)
Fresnel_coeffs = Fresnel_equations(89*np.pi/180, E)

fig, ax = plt.subplots()
ax.plot(E, Fresnel_coeffs.Rs, label='Rs')
ax.plot(E, Fresnel_coeffs.Rp, label='Rp')
ax.set_xlabel('Energy [keV]')
ax.legend()

alpha = np.arange(0, 10*60, 1)/60*np.pi/180
Fresnel_coeffs = Fresnel_equations(alpha, 1)

fig, ax = plt.subplots()
ax.plot(alpha, Fresnel_coeffs.Rs, label='Rs')
ax.plot(alpha, Fresnel_coeffs.Rp, label='Rp')
ax.set_xlabel(r'$\alpha$ [rad]')
ax.legend()

# +
R = (np.abs(Fresnel_coeffs.Rs)**2 + np.abs(Fresnel_coeffs.Rp)**2)/2

fig, ax = plt.subplots()
ax.plot(alpha, R)
ax.set_xlabel(r'$\alpha$ [rad]')
# -

# pd.DataFrame({'angolo':alpha*180/np.pi*60, 'R':R}).to_csv('/Users/john/Desktop/R.csv')

# # Multilayer

reflection_R_many_layers(0.01*np.pi/180, 8, [1000], [None, os.path.join(datapath, 'Au.nk'), os.path.join(datapath, 'Ni.nk')])

reflection_R_many_layers(0.01*np.pi/180, 8, [100, 1000], [None, os.path.join(datapath, 'Cr.nk'), os.path.join(datapath, 'Au.nk'), os.path.join(datapath, 'Ni.nk')])


fig, ax = plt.subplots()
angles = np.linspace(0, 10, 10000)/180*np.pi
ax.plot(angles,
       reflection_R_many_layers(angles, 8.05, [300], [None, os.path.join(datapath, 'Au.nk'), os.path.join(datapath, 'Ni.nk')]).Rs, label='Au [1000A] + Ni')
ax.plot(angles,
       reflection_R_many_layers(angles, 8, [100, 1000], [None, os.path.join(datapath, 'C.nk'), os.path.join(datapath, 'Au.nk'), os.path.join(datapath, 'Ni.nk')]).Rs,
       label='C [100A] + Au [1000A] + Ni')
ax.set_yscale('log')
ax.set_xlabel('Angle [rad]')
ax.set_ylabel('Rs')
ax.legend()

fig, ax = plt.subplots()
angles = np.linspace(0, 4000, 10000)/(180/np.pi*3600)
coeffs = reflection_R_many_layers(angles, 8.05, [302], [None, os.path.join(datapath, 'Au.nk'), os.path.join(datapath, 'a-SiO2.nk')])
ax.plot(angles*180/np.pi*3600, coeffs.Rs**2, label='Rs')
ax.plot(angles*180/np.pi*3600, coeffs.Rp**2, label='Rp')
ax.set_yscale('log')
ax.set_xlabel('Angle [rad]')
ax.legend()


# # Multilayer receipt

# + active=""
# d_list, material_nk_files_list = multilayers_from_receipt(high_Z_path=os.path.join(datapath, 'W.nk'), 
#                                                           low_Z_path= os.path.join(datapath, 'Si.nk'), 
#                                                           bottom_path=os.path.join(datapath, 'a-SiO2.nk'), 
#                                                           N=445
#                                                           , a=22.560100, b=20.091492, c=0.074198119, G=0.37473476)
# -

d_list, material_nk_files_list = multilayers_from_receipt(high_Z_path=os.path.join(datapath, 'W.nk'), 
                                                          low_Z_path= os.path.join(datapath, 'Si.nk'), 
                                                          bottom_path=os.path.join(datapath, 'a-SiO2.nk'), 
                                                          N=98
                                                          , a=73.736034, b=-0.98098659, c=0.17746457, G=0.44570398)

energies = np.linspace(20, 25, 1000)
coeffs = reflection_R_many_layers(0.22*np.pi/180, energies, d_list, material_nk_files_list)

# + active=""
# #pure Au comparison
# energies = np.linspace(20, 25, 1000)
# coeffs = generate_Fresnel_coefficients(None, os.path.join(datapath, 'Au.nk'))(0.22*np.pi/180, energies)
# -

fig, ax = plt.subplots()
ax.plot(energies, np.abs(coeffs.Rs)**2, label=r'$R_s^2$')
ax.plot(energies, np.abs(coeffs.Rp)**2, label=r'$R_p^2$')
ax.plot(energies, (np.abs(coeffs.Rs)**2 + np.abs(coeffs.Rp)**2)/2, label=r'$\frac{R_s^2+R_p^2}{2}$')
ax.legend()
ax.set_xlabel('E [keV]')

plt.show()
