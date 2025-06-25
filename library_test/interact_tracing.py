import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from PyQt5 import QtCore, uic
from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication, QLabel, QComboBox, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QCheckBox, QSizePolicy, QGroupBox, QButtonGroup, QRadioButton, QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT

import os
# sys.path.append('..')
# sys.path.append('.')
from xJtracing.Wolter_I import simulate_a_WolterI
from xJtracing.Lobster_geometry import define_Lobster_mirrors, generate_cells_arranged_variable_L
from xJtracing.Lobster_ray_tracing import simulate_an_Angel, simulate_a_Schmidt_or_KB
from xJtracing.KB import define_KB_mirrors, generate_KB_cells_adaptive_side
from xJtracing.benchmarks import bins_around_mean
from xJtracing.tracing_utils import save_telescope_pars, load_telescope_pars
from xJtracing.data.paths import path_erosita, path_nk

Au_path = os.path.join(path_nk, 'Au.nk')


def eROSITA_pars():
    """
    Geometric telescope parameters of an eROSITA module.
    """

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
            'L1s':L1, 'f0':focal_length, 'best_focal_plane':focal_length, 'inner_mirror':inner_mirror}
    return telescope_pars


def Lobster_pars(configuration, radius_top, cell_side, cell_length, cell_distance, number_cells_per_side, radius_bottom=None):
    """
    Geometric parameters of a Lobster telescope.
    """
    if configuration=='Angel':
        telescope_pars = define_Lobster_mirrors(configuration='Angel', reflecting='total', radius=radius_top, cell_side=cell_side, 
                                        cell_length=cell_length, cell_distance=cell_distance, 
                                                number_cells_per_side=number_cells_per_side, max_reflections=2)
    elif configuration=="Schmidt":
        telescope_pars = {}
        telescope_pars['top'] = define_Lobster_mirrors(configuration='Schmidt_top', reflecting='total', radius=radius_top, cell_side=cell_side, 
                                        cell_length=cell_length, cell_distance=cell_distance, number_cells_per_side=number_cells_per_side, max_reflections=2,
                                           radius_inner=radius_bottom, complete_vectorization=False
                                           )

        telescope_pars['bottom'] = define_Lobster_mirrors(configuration='Schmidt_bottom', reflecting='total', radius=radius_bottom, cell_side=cell_side, 
                                        cell_length=cell_length, cell_distance=cell_distance, number_cells_per_side=number_cells_per_side, max_reflections=2, 
                                               complete_vectorization=False)
        
    return telescope_pars

def KB_pars(radius_top, radius_bottom, cell_side, cell_length, cell_distance, number_cells_per_side):
    """
    Geometric parameters of a Kirkpatrick-Baez telescope.
    """
    telescope_pars = {}
    telescope_pars['top'] = define_KB_mirrors('KB_top', cell_distance, number_cells_per_side, cell_side, cell_length, radius_top, radius_bottom, 
                                           complete_vectorization=False, cells_arrengement_function=generate_KB_cells_adaptive_side)
    telescope_pars['bottom'] = define_KB_mirrors('KB_bottom', cell_distance, number_cells_per_side, cell_side, cell_length, radius_bottom, 
                                              complete_vectorization=False, cells_arrengement_function=generate_KB_cells_adaptive_side)
    return telescope_pars



def run_WolterI_simulation_interactive(telescope_pars, off_axis_angle_deg, pa_deg, optimize_focal_plane, rays_in_mm2, ax_image, ax_section):
    """
    Main function for simulating a Wolter-I.
    """

    Wolter_I_out = simulate_a_WolterI(off_axis_angle_deg, pa_deg, telescope_pars,  
                                  material_nk_files_list=[Au_path], 
                        d_list=[], rugosity_hew=False, optimize_focal_plane=optimize_focal_plane, 
                                  # apply_tilt={'tilt_deg':0.5, 'pa_deg':0, 'xshift':None, 'yshift':None},
                                  plot_tracing=ax_section, rays_in_mm2=rays_in_mm2
                                 )

    result_string = f"""Aeff = {Wolter_I_out['Aeff']/100:.1f}cm²
HEW = {Wolter_I_out['hew']:.1f}arcsec
Best focal plane = {Wolter_I_out['best_focal_plane']:.1f}mm
Started with {Wolter_I_out['nrays_initial']:,} rays
Finished with {Wolter_I_out['x'].size:,} rays"""

    x, y = Wolter_I_out['x'], Wolter_I_out['y']
    ax_image.hist2d(x, y, bins_around_mean(x, y, half_space = 2, increment = 0.1))
    return result_string
    

def run_Lobster_simulation_interactive(configuration, telescope_pars, off_axis_angle_deg, pa_deg, rays_in_mm2, ax_image, ax_section):
    """
    Main function for simulating a celluler optics module.
    """
    if configuration=='Angel':
        telescope_out = simulate_an_Angel(off_axis_angle_deg, pa_deg, telescope_pars, material_nk_files_list=[Au_path], 
                        d_list=[], rugosity_hew=False, rays_in_mm2=rays_in_mm2, plot_tracing=ax_section)
    elif configuration=="Schmidt":
        telescope_out = simulate_a_Schmidt_or_KB(off_axis_angle_deg, pa_deg, telescope_pars['top'], telescope_pars['bottom'], material_nk_files_list=[Au_path], 
                                   d_list=[], rugosity_hew=False, rays_in_mm2=rays_in_mm2, complete_vectorization=False, plot_tracing=ax_section)
    elif configuration=='KB':
        telescope_out = simulate_a_Schmidt_or_KB(off_axis_angle_deg, pa_deg, telescope_pars['top'], telescope_pars['bottom'], material_nk_files_list=[Au_path], 
                                   d_list=[], rugosity_hew=False, rays_in_mm2=rays_in_mm2, design='KB', complete_vectorization=False, plot_tracing=ax_section)
    
    result_string = f"""Aeff = {telescope_out['Aeff']/100:.1f}cm²
HEW = {telescope_out['hew']:.1f}arcsec
Started with {telescope_out['nrays_initial']:,} rays
Finished with {telescope_out['x'].size:,} rays
Fractions of rays:
• 0 reflections: {telescope_out['hew_dict']['fraction0']:.2f}
• 1 reflections: {telescope_out['hew_dict']['fraction1']:.2f}
• 2 reflections in center: {telescope_out['hew_dict']['fraction_center']:.2f}
• 2 reflections in outer clouds: {telescope_out['hew_dict']['fraction_clouds']:.2f}
"""
    
    x, y = telescope_out['x'], telescope_out['y']
    ax_image.hist2d(x, y, bins_around_mean(x, y, half_space = 50, increment = 0.1), norm="log")
    return result_string


class NavigationToolbar_small(NavigationToolbar2QT):
    '''Only display the buttons we need to save space.'''
    toolitems = [t for t in NavigationToolbar2QT.toolitems if
                 t[0] in ( 'Pan', 'Zoom', 'Back', 'Forward', 'Subplots', 'Save')]



class MplCanvas(FigureCanvas):
    """
    Class to make plots.
    """
    def __init__(self, double_ax):
        if double_ax:
            fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        else:
            fig, ax = plt.subplots()
        self.ax = ax
        self.fig = fig
        FigureCanvas.__init__(self, self.fig)
        # self.setFocusPolicy(QtCore.Qt.StrongFocus)  # Ensures widget can receive focus
        plt.draw()
        self.draw()


class MplWidgetMain(QWidget):
    """
    Class to make plots.
    """
    def __init__(self, double_ax=False, parent = None, reflex=None):
        QWidget.__init__(self, parent)
        self.setParent(parent)
        self.canvas = MplCanvas(double_ax)
        vbl = QVBoxLayout()
        vbl.addWidget(self.canvas)
        self.ntb = NavigationToolbar2QT(self.canvas, self)
        vbl.addWidget(self.ntb)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.draw()
        self.setLayout(vbl)


class WolterI_interface(QWidget):
    """
    Interface for Wolter-I geometrical parameters selection (which currently defaults to eROSITA not allowing other choises from the interface, only from the load file).
    """
    def __init__(self):
        super().__init__()
        layoutV = QVBoxLayout()
        label = QLabel("Geometrical parameters are these from a module of eROSITA.", self)
        layoutV.addWidget(label)
        self.setLayout(layoutV)
    

class Lobster_interface(QWidget):
    """
    Interface to select geometrical parameters of a cellular optics module.
    """
    def __init__(self, configuration):
        super().__init__()
        self.inputs = {}
        self.layoutV = QVBoxLayout()

        
        if configuration=="Angel":
            self._add_field("Radius [mm]", "5000")
        elif configuration in ["Schmidt", "KB"]:
            self._add_field("Radius top [mm]", "5000")
            self._add_field("Radius bottom [mm]", "4800")
        self._add_field("cell side [mm]", "0.75")
        self._add_field("cell length [mm]", "75")
        self._add_field("cell distance [mm]", "1.1")
        self._add_field("Number of cells per side", "200")

        self.setLayout(self.layoutV)

    def _add_field(self, label, default):
        field = QHBoxLayout()
        field_label = QLabel(label+" = ", self)
        self.inputs[label] = QLineEdit(field_label)
        self.inputs[label].setText(default)
        field.addWidget(field_label)
        field.addWidget(self.inputs[label])
        self.layoutV.addLayout(field)
        

class mainSimulationWindow(QWidget):
    """
    Main simulatin window with input geometrical parameters (loading another widget), input rays parameters and output.
    """
    def __init__(self, design_selection):
        super().__init__()

        self.design_selection = design_selection
        
        self.setWindowTitle(self.design_selection)
        self.setGeometry(100, 100, 1400, 700)
        layoutV = QVBoxLayout()
        button_group = QButtonGroup(self)
        
        load_option = QVBoxLayout()
        self.load_option_radio = QRadioButton("Use Loaded Configuration", self)
        button_group.addButton(self.load_option_radio)
        self.load_button = QPushButton("Load Configuration", self)
        self.load_button.clicked.connect(self.load_configuration)
        load_option.addWidget(self.load_option_radio)
        load_option.addWidget(self.load_button)
        layoutV.addLayout(load_option)

        save_option = QVBoxLayout()
        self.save_option_radio = QRadioButton("Define Here Configuration", self)
        button_group.addButton(self.save_option_radio)
        self.save_button = QPushButton("Save Defined Configuration", self)
        self.save_button.clicked.connect(self.save_configuration)
        save_option.addWidget(self.save_option_radio)
        save_option.addWidget(self.save_button)
        layoutV.addLayout(save_option)
        self.save_option_radio.setChecked(True)

        if self.design_selection == "Wolter-I eROSITA":
            self.design_specific_section = WolterI_interface()
        elif self.design_selection == "Lobster-Angel":
            self.design_specific_section = Lobster_interface('Angel')
        elif self.design_selection == "Lobster-Schmidt":
            self.design_specific_section = Lobster_interface('Schmidt')
        elif self.design_selection == "Kirkpatrick-Baez":
            self.design_specific_section = Lobster_interface('KB')
        layoutV.addWidget(self.design_specific_section)
        
        off_axis = QHBoxLayout()
        off_axis_label = QLabel("Off-axis angle [deg]", self)
        self.off_axis_input = QLineEdit(self)
        self.off_axis_input.setText("0")
        off_axis.addWidget(off_axis_label)
        off_axis.addWidget(self.off_axis_input)
        layoutV.addLayout(off_axis)

        pa_off_axis = QHBoxLayout()
        pa_off_axis_label = QLabel("Off-axis position angle [deg]", self)
        self.pa_off_axis_input = QLineEdit(self)
        self.pa_off_axis_input.setText("0")
        pa_off_axis.addWidget(pa_off_axis_label)
        pa_off_axis.addWidget(self.pa_off_axis_input)
        layoutV.addLayout(pa_off_axis)
        
        rays_in_mm2 = QHBoxLayout()
        rays_in_mm2_label = QLabel(r"Input rays density [1/mm²]", self)
        self.rays_in_mm2_input = QLineEdit(self)
        self.rays_in_mm2_input.setText("10")
        rays_in_mm2.addWidget(rays_in_mm2_label)
        rays_in_mm2.addWidget(self.rays_in_mm2_input)
        layoutV.addLayout(rays_in_mm2)

        optimize_focal_plane = QHBoxLayout()
        optimize_focal_plane_label = QLabel("Optimize focal plane", self)
        self.optimize_focal_plane_input = QCheckBox(self)
        if self.design_selection not in ["Wolter-I eROSITA"]:
            self.optimize_focal_plane_input.setEnabled(False)
        optimize_focal_plane.addWidget(optimize_focal_plane_label)
        optimize_focal_plane.addWidget(self.optimize_focal_plane_input)
        layoutV.addLayout(optimize_focal_plane)

        plot_section = QHBoxLayout()
        plot_section_label = QLabel("Plot Section", self)
        self.plot_section_input = QCheckBox(self)
        plot_section.addWidget(plot_section_label)
        plot_section.addWidget(self.plot_section_input)
        layoutV.addLayout(plot_section)
        
        self.simulate_button = QPushButton("Simulate", self)
        self.simulate_button.clicked.connect(self.simulate)
        layoutV.addWidget(self.simulate_button)

        group_box = QGroupBox("", self)
        group_layout = QVBoxLayout()
        self.label_out = QLabel("""""", self)
        group_layout.addWidget(self.label_out)
        group_box.setLayout(group_layout)        
        layoutV.addWidget(group_box)
        
        
        self.layoutH = QHBoxLayout()
        self.layoutH.addLayout(layoutV)
        
        self.plot_image = MplWidgetMain()
        self.layoutH.addWidget(self.plot_image)

        self.already_add_section_plot = False

        self.setLayout(self.layoutH)

    def generate_telescope_pars(self):
        """
        Load the geometrical parameters from other functions.
        """
        
        def get_lobster_key(label):
            return float(self.design_specific_section.inputs[label].text())
            
        if self.design_selection == "Wolter-I eROSITA":
            self.telescope_pars = eROSITA_pars()
        elif self.design_selection == "Lobster-Angel":
            input_list = list(map(get_lobster_key, ["Radius [mm]", "cell side [mm]", "cell length [mm]", "cell distance [mm]", "Number of cells per side"]))
            self.telescope_pars = Lobster_pars("Angel", *input_list)
        elif self.design_selection == "Lobster-Schmidt":
            input_list = list(map(get_lobster_key, ["Radius top [mm]", "cell side [mm]", "cell length [mm]", "cell distance [mm]", "Number of cells per side", "Radius bottom [mm]"]))
            self.telescope_pars = Lobster_pars("Schmidt", *input_list)
        elif self.design_selection == "Kirkpatrick-Baez":
            input_list = list(map(get_lobster_key, ["Radius top [mm]", "Radius bottom [mm]", "cell side [mm]", "cell length [mm]", "cell distance [mm]", "Number of cells per side"]))
            self.telescope_pars = KB_pars(*input_list)


    def load_configuration(self):
        """
        Open a file dialog to select the configuration file to load.
        """
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Configuration File")
        self.telescope_pars = load_telescope_pars(file_name)
    
    def save_configuration(self):
        """
        Open a file dialog to select where to save the configuration.
        """
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Configuration File")
        if self.save_option_radio.isChecked():
            self.generate_telescope_pars()
        save_telescope_pars(self.telescope_pars, file_name)


    def simulate(self):
        """
        Simulate and plot.
        """
        self.plot_image.canvas.ax.clear()
        self.plot_image.canvas.ax.set_xlabel('x [mm]')
        self.plot_image.canvas.ax.set_ylabel('y [mm]')
        if self.plot_section_input.isChecked():
            if self.already_add_section_plot == False:
                if self.design_selection in ["Wolter-I eROSITA"]:
                    self.plot_section = MplWidgetMain()
                    self.plot_section.canvas.ax.clear()
                else:
                    self.plot_section = MplWidgetMain(double_ax=True)
                    self.plot_section.canvas.ax[0].clear()
                    self.plot_section.canvas.ax[1].clear()
                self.layoutH.addWidget(self.plot_section)
                self.already_add_section_plot = True
            
        elif self.plot_section_input.isChecked() == False and self.already_add_section_plot:
            self.layoutH.removeWidget(self.plot_section)
            self.plot_section.deleteLater()
            self.already_add_section_plot = False
        
        off_axis_angle_deg = float(self.off_axis_input.text())
        pa_deg = float(self.pa_off_axis_input.text())
        optimize_focal_plane = self.optimize_focal_plane_input.isChecked()
        rays_in_mm2 = float(self.rays_in_mm2_input.text())

        if self.save_option_radio.isChecked():
            self.generate_telescope_pars()

        ax_image = self.plot_image.canvas.ax
        ax_section = self.plot_section.canvas.ax if self.plot_section_input.isChecked() else False
        
        if self.design_selection == "Wolter-I eROSITA":
            outtext = run_WolterI_simulation_interactive(self.telescope_pars, off_axis_angle_deg, pa_deg, optimize_focal_plane, rays_in_mm2, ax_image, ax_section)
        elif self.design_selection == "Lobster-Angel":
            outtext = run_Lobster_simulation_interactive("Angel", self.telescope_pars, off_axis_angle_deg, pa_deg, rays_in_mm2, ax_image, ax_section)
        elif self.design_selection == "Lobster-Schmidt":
            outtext = run_Lobster_simulation_interactive("Schmidt", self.telescope_pars, off_axis_angle_deg, pa_deg, rays_in_mm2, ax_image, ax_section)
        elif self.design_selection == "Kirkpatrick-Baez":
            outtext = run_Lobster_simulation_interactive("KB", self.telescope_pars, off_axis_angle_deg, pa_deg, rays_in_mm2, ax_image, ax_section)
        
        self.plot_image.canvas.fig.tight_layout()
        self.plot_image.canvas.draw()
        
        self.label_out.setText(outtext)
        
        if self.plot_section_input.isChecked():
            self.plot_section.canvas.fig.tight_layout()
            self.plot_section.canvas.draw()
            



class SelectModelWindow(QWidget):
    """
    When opening the application, select the model.
    """
    
    def __init__(self, reflex=None):
        super().__init__()
        self.setWindowTitle("Select Optical Design")
        # self.setGeometry(100, 100, 800, 600)

        self.label = QLabel("""Welcome to the interactive side of xJtracing. 
Please select your favorite optical design to simulate.
        """, self)

        self.combo_box = QComboBox(self)
        self.combo_box.addItem("Please select")
        self.combo_box.addItem("Wolter-I eROSITA")
        self.combo_box.addItem("Lobster-Angel")
        self.combo_box.addItem("Lobster-Schmidt")
        self.combo_box.addItem("Kirkpatrick-Baez")

        self.combo_box.currentIndexChanged.connect(self.menu_select_design)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.combo_box)

        self.setLayout(layout)


    def menu_select_design(self):
        # Close the current window (first window)
        self.close()

        self.second_window = mainSimulationWindow(self.combo_box.currentText())
        self.second_window.show()


if __name__=='__main__':
    app = QApplication(sys.argv)
    window = SelectModelWindow()
    window.show()
    sys.exit(app.exec_())