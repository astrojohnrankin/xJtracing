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
# sys.path.append('..')
# sys.path.append('.')

from xJtracing.Lobster_ray_tracing import simulate_an_Angel, simulate_a_Schmidt_or_KB
from xJtracing.plotting import plot_rays_map, plot_image
from xJtracing.Lobster_geometry import define_Lobster_mirrors, generate_cells_arranged_variable_L

import numpy as np
import matplotlib.pyplot as plt

try:
    Au_path = os.path.basename(os.path.dirname(__file__))+'/../xJtracing/data/nk/Au.nk'
except:
    Au_path = '../xJtracing/data/nk/Au.nk'
# -

off_axis_angle_deg, pa_deg = 0.0, 0.0

complete_vectorization = False

# + active=""
# complete_vectorization = True

# +
radius_top, radius_bottom = 5000, 4800

telescope_pars_top = define_Lobster_mirrors(configuration='Schmidt_top', reflecting='total', radius=radius_top, cell_side=0.75, 
                                        cell_length=75, cell_distance=1.1, number_cells_per_side=200, max_reflections=2,
                                           radius_inner=radius_bottom, complete_vectorization=complete_vectorization
                                            # , cells_arrengement_function=generate_cells_arranged_variable_L
                                           )

telescope_pars_bottom = define_Lobster_mirrors(configuration='Schmidt_bottom', reflecting='total', radius=radius_bottom, cell_side=0.75, 
                                        cell_length=75, cell_distance=1.1, number_cells_per_side=200, max_reflections=2, 
                                               complete_vectorization=complete_vectorization
                                              # , cells_arrengement_function=generate_cells_arranged_variable_L
                                              )
# -

telescope_out = simulate_a_Schmidt_or_KB(off_axis_angle_deg, pa_deg, telescope_pars_top, telescope_pars_bottom, material_nk_files_list=[Au_path], 
                                   d_list=[], rugosity_hew=False, rays_in_mm2=3
                                         # , plot_tracing = True
                                        )


telescope_out['Aeff']/100

telescope_out

plot_rays_map(telescope_out['rays_maps'], rays_keys=[0, '1up', '1down', 2, '3+'])

plot_image(telescope_out['rays_maps']['total']['x'], telescope_out['rays_maps']['total']['y'], '', '/Users/john/Desktop/', padding=100, pixel_size=0.1)


plt.show()
