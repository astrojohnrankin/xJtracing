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
import sys
import os
sys.path.append('..')
sys.path.append('.')

from xJtracing.Lobster_ray_tracing import simulate_an_Angel, create_Lobster_mirrors
from xJtracing.plotting import plot_rays_map, plot_image, plot_mirror
from xJtracing.Lobster_geometry import define_Lobster_mirrors, generate_cells_arranged_variable_L

import numpy as np
import matplotlib.pyplot as plt

try:
    Au_path = os.path.basename(os.path.dirname(__file__))+'/../xJtracing/data/nk/Au.nk'
except:
    Au_path = '../xJtracing/data/nk/Au.nk'
# -

off_axis_angle_deg, pa_deg = 0, 45

telescope_pars = define_Lobster_mirrors(configuration='Angel', reflecting='total', radius=5000, cell_side=0.75, 
                                        cell_length=75, cell_distance=1.1, number_cells_per_side=200, max_reflections=2)

telescope_pars

Lobster_out = simulate_an_Angel(off_axis_angle_deg, pa_deg, telescope_pars, material_nk_files_list=[Au_path], 
                        d_list=[], rugosity_hew=False,
                               # ,plot_tracing=True, 
                                rays_in_mm2=20
                               )

Lobster_out

Lobster_out['Aeff']/100

plot_rays_map(Lobster_out['rays_maps'])

plot_image(Lobster_out['rays_maps']['total']['x'], Lobster_out['rays_maps']['total']['y'], '', '/Users/john/Desktop/', padding=100, pixel_size=0.1)


plt.show()
