# %load_ext autoreload
# %autoreload 2

# %matplotlib widget

# +
import os
import sys
# sys.path.append('..')
# sys.path.append('.')
from xJtracing.Wolter_I import simulate_a_WolterI

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

try:
    Au_path = os.path.basename(os.path.dirname(__file__))+'/../xJtracing/data/nk/Au.nk'
    path_erosita = os.path.basename(os.path.dirname(__file__))+'/../xJtracing/data/eROSITA_design_UNCERTAIN.mms'
except:
    Au_path = '../xJtracing/data/nk/Au.nk'
    path_erosita = '../xJtracing/data/eROSITA_design_UNCERTAIN.mms'
# -

off_axis_angle_deg, pa_deg = 0.1, 0

# +
tab_eRosita = pd.read_csv(path_erosita, sep='\s+')
spessori = tab_eRosita['thk(mm)'].values
weights = tab_eRosita['m(kg)'].values
radii_c = 0.5*np.array([348.483, 338.522, 328.799, 319.406, 310.232, 301.378, 292.733, 284.394, 276.251, 268.401, 260.738, 253.256, 246.049, 239.015, 232.149, 225.542, 219.094, 212.800, 206.752, 200.850, 195.086, 189.461, 184.067, 178.801, 173.661, 168.744, 163.940, 159.253, 154.675, 150.208, 145.944, 141.783, 137.719, 133.754, 129.881, 126.202, 122.606, 119.099, 115.674, 112.331, 109.166, 106.077, 103.060, 100.117, 97.242, 94.435, 91.694, 89.119, 86.607, 84.152, 81.751, 79.350, 
                      76.949, 74.549])
radii_max =  0.5*np.array([356.528, 346.338, 336.391, 326.782, 317.401, 308.342, 299.499, 290.966, 282.637, 274.607, 266.766, 259.112, 251.741, 244.545, 237.518, 230.760, 224.164, 217.724, 211.538, 205.498, 199.602, 193.846, 188.328, 182.940, 177.681, 172.649, 167.735, 162.940, 158.256, 153.685, 149.323, 145.066, 140.909, 136.851, 132.890, 129.123, 125.447, 121.858, 118.353, 114.932, 111.695, 108.534, 105.449, 102.436, 99.495, 96.622, 93.819, 91.184, 88.613, 86.100, 83.646, 81.189, 
                      78.732, 76.275])
r_inner = np.append(radii_c[1:]+spessori[1:], 0.001)

focal_length = 1600
L = 150

L1 = np.repeat(L, radii_max.size)
inner_mirror = True
telescope_pars = {'radii_parabola':radii_max, 'radii_center':radii_c, 'radii_center_inner':r_inner, 
        'L1s':L1, 'f0':focal_length, 'inner_mirror':inner_mirror}


# + active=""
# telescope_pars = generator_f_wolterI_auto(R_initial=35, squared_size=220, f0=2500, L=150, thickness=0.35, inner_mirror=True)
# -

telescope_pars

Wolter_I_out = simulate_a_WolterI(off_axis_angle_deg, pa_deg, telescope_pars,  
                                  material_nk_files_list=[Au_path], 
                        d_list=[], rugosity_hew=False, optimize_focal_plane=True, 
                                  plot_tracing=True, rays_in_mm2=10
                                 )

Wolter_I_out

fig, ax = plt.subplots()
ax.hist2d(Wolter_I_out['x'], Wolter_I_out['y'], np.arange(-30,30,0.01))

fig, ax = plt.subplots()
ax.plot(Wolter_I_out['x'], Wolter_I_out['y'], '.')
ax.set_aspect('equal')

plt.show()
