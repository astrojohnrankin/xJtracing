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

from xJtracing.KB import define_KB_mirrors, generate_KB_cells_equal, generate_KB_cells_adaptive_L, generate_KB_cells_adaptive_side, create_KB_mirrors
from xJtracing.Lobster_ray_tracing import convert_telescope_pars_to_list, simulate_ortogonal_system, simulate_a_Schmidt_or_KB
from xJtracing.generators import create_square_rays_for_Schmidt
from xJtracing.plotting import plot_rays_map, plot_image

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
cell_side=0.75
cell_length=75
cell_distance=1.1
number_cells_per_side=200

telescope_pars_top = define_KB_mirrors('KB_top', cell_distance, number_cells_per_side, cell_side, cell_length, radius_top, radius_bottom, 
                                       complete_vectorization=complete_vectorization, cells_arrengement_function=generate_KB_cells_adaptive_side)
telescope_pars_bottom = define_KB_mirrors('KB_bottom', cell_distance, number_cells_per_side, cell_side, cell_length, radius_bottom, 
                                          complete_vectorization=complete_vectorization, cells_arrengement_function=generate_KB_cells_adaptive_side)

# -

telescope_out = simulate_a_Schmidt_or_KB(off_axis_angle_deg, pa_deg, telescope_pars_top, telescope_pars_bottom, material_nk_files_list=[Au_path], 
                                   d_list=[], rugosity_hew=False, rays_in_mm2=1.5, design='KB', complete_vectorization=complete_vectorization
                                        ,plot_tracing=True
                                        )


telescope_out['Aeff']/100

telescope_out

plot_rays_map(telescope_out['rays_maps'], rays_keys=[0, '1up', '1down', 2, '3+'])

plot_image(telescope_out['rays_maps']['total']['x'], telescope_out['rays_maps']['total']['y'], '', '/Users/john/Desktop/', padding=100, pixel_size=0.1)

plt.show()
