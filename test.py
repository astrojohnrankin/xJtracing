import unittest

import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

from xJtracing.absorption import generate_refractive_index_function, generate_Fresnel_coefficients, reflection_R_many_layers
from xJtracing.benchmarks import half_energy_diameter
from xJtracing.rays import ray_versor, generate_ray_direction_equations
from xJtracing.mirror import generate_Wolter_parabola, generate_KB_parabola, generate_Wolter_iperbola, generate_flat_surface
from xJtracing.intersection import generate_parabola_intersection_function, generate_parabola_KB_intersection_function, generate_iperbola_intersection_function, generate_flat_surface_intersection_function, create_image
from xJtracing.reflections import find_tangent_plane_and_normal, find_normal_to_flat_plane, reflect_ray


class TestAbsorption(unittest.TestCase):
    def test_absorption(self):
        datapath = 'xJtracing/data/nk'
        
        E = np.array([0.1, 1, 8])
        
        n = generate_refractive_index_function(os.path.join(datapath, 'Au.nk'))
        np.testing.assert_almost_equal(n(E), np.array([0.91446021+3.52126354e-02j, 
                                                       0.99789407+1.03049182e-03j,
                                                       0.99995227+4.96100298e-06j]))
        
        Fresnel_equations = generate_Fresnel_coefficients(None, os.path.join(datapath, 'Au.nk'))
        Fresnel_coeffs = Fresnel_equations(89*np.pi/180, E)
        np.testing.assert_almost_equal(Fresnel_coeffs.Rs, 
                                        np.array([4.43420960e-02-1.92152590e-02j,
                                                  1.05413222e-03-5.16490543e-04j,
                                                  2.38736813e-05-2.48137576e-06j]))
        np.testing.assert_almost_equal(Fresnel_coeffs.Rp, 
                                        np.array([4.43130931e-02-1.92013418e-02j, 
                                                  1.05348905e-03-5.16174583e-04j,
                                                  2.38591374e-05-2.47986403e-06j]))
        
        multy_reflection = reflection_R_many_layers(0.01*np.pi/180, E, [100, 1000], 
                                [None, os.path.join(datapath, 'Cr.nk'), 
                                 os.path.join(datapath, 'Au.nk'), 
                                 os.path.join(datapath, 'Ni.nk')])
        np.testing.assert_almost_equal(multy_reflection.Rs, np.array([-0.99976846-0.00094406j, -0.99872008-0.00646608j, -0.99589159-0.05288352j]))
        np.testing.assert_almost_equal(multy_reflection.Rp, np.array([0.99973902+0.00082224j, 0.99871668+0.00644793j, 0.9958916 +0.05288123j]))


class TestBenchmarks(unittest.TestCase):    
    def test_benchmarks(self):
        x = np.array([1.1, 1.3, 1.0])
        y = np.array([1.0, 1.5, 1.1])
        hew = half_energy_diameter(x, y, 1000)
        np.testing.assert_almost_equal(hew, 83.6439889839373)



class TestWolter(unittest.TestCase):    
    def test_Wolter(self):   
        e = ray_versor(rho=0.0001, theta=10*np.pi/180)
        np.testing.assert_array_almost_equal(e, 
            (9.848077513708618e-05, 1.7364817737751674e-05, 0.999999995))
        x_0 = [50.52, 0, 100]
        x_func, y_func = generate_ray_direction_equations(e=e, x_0=x_0)
        R0 = 50.5
        theta = np.arctan2(50.5, 1600)/4
        L1 = 150
        fx, fy, parabola = generate_Wolter_parabola(R0, theta)
        fx, fy, iperbola = generate_Wolter_iperbola(R0, theta*3, theta, 1600)
        parabola_intersection_f = generate_parabola_intersection_function(R0, theta, 0, L1)
        (x_intersect, y_intersect, z_intersect), exists_intersection = parabola_intersection_f(x_0, e)
        np.testing.assert_almost_equal((x_intersect, y_intersect, z_intersect, exists_intersection), 
                                      (50.51028028123062, -0.0017138486639411959, 1.3033893441916007, np.array(True)))
        tan_plane_z, n_normal = find_tangent_plane_and_normal(parabola, x_intersect, y_intersect)   
        np.testing.assert_almost_equal(n_normal, [ 9.99968902e-01, -3.37604339e-05, -7.88631790e-03])
        e_reflected1 = reflect_ray(e, n_normal, x_intersect, y_intersect, z_intersect)
        iperbola_intersection_f = generate_iperbola_intersection_function(R0, 3*theta, theta, 1600, -L1, 0)
        (x_intersect, y_intersect, z_intersect), exists_intersection = iperbola_intersection_f((x_intersect, y_intersect, z_intersect), e_reflected1)
        tan_plane_z, n_normal2 = find_tangent_plane_and_normal(iperbola, x_intersect, y_intersect)
        e_reflected2 = reflect_ray(e_reflected1, n_normal2, x_intersect, y_intersect, z_intersect)
        e_reflected2_tuple_of_array = list(map(lambda _item: np.array([_item]), e_reflected2))
        x_det, y_det, z_det = create_image(e_reflected2_tuple_of_array, (x_intersect, y_intersect, z_intersect), 1600)
        np.testing.assert_almost_equal((x_det, y_det, z_det[0]), 
                                      (np.array([-0.15772045]), np.array([-0.02780149]), np.array([-1600])))


class TestFlat(unittest.TestCase):
    def test_flat(self):
        e = ray_versor(rho=25*np.pi/180, theta=0)
        x0_ray = [0,0,0]
    
        alpha = 0.001
        theta = 0
        z_low, z_up = -10, 10
        bounds_x = lambda x: [-np.inf, np.inf]
        bounds_y = lambda y: [-10, 10]
        
        equation_x, equation_y = generate_flat_surface(alpha=alpha, theta=theta, x0=0, y0=0, z0=0)
        
        np.testing.assert_almost_equal(equation_x(1, 0), 0.0010000003333334668)
        
        intersection_function = generate_flat_surface_intersection_function(alpha=alpha, 
                                                            theta=theta, mirror_x0=0, mirror_y0=0, 
                                                            mirror_z0=0, z_low=z_low, z_up=z_up, 
                                                            bounds_x=bounds_x, bounds_y=bounds_y)
        
        (x_intersect, y_intersect, z_intersect), exists_intersection = intersection_function(x0_ray, e)
        (x_intersect, y_intersect, z_intersect), exists_intersection
        
        np.testing.assert_almost_equal([x_intersect, y_intersect, z_intersect], [0,0,0])
        assert exists_intersection==True
        
        n_normal = find_normal_to_flat_plane(alpha, theta, x_intersect, y_intersect, z_intersect)
        
        np.testing.assert_almost_equal(n_normal, [ 0.9999995,  0.       , -0.001    ])
        
        e_reflected = reflect_ray(e, n_normal, x_intersect, y_intersect, z_intersect)
        np.testing.assert_almost_equal(e_reflected, [-0.4208048 ,  0.        ,  0.90715121])
        
                
class Test_KB_parabola(unittest.TestCase):    
    def test_KB_parabola(self):   
        e = ray_versor(rho=0.0001, theta=0.0001*np.pi/180)
        x_0 = [0, 50, 100]
        x_func, y_func = generate_ray_direction_equations(e=e, x_0=x_0)

        f, L, R0, axis = 20, 100, 50, 'y'
        z_low, z_up = -1000, 1000
        bounds_x = [-1000, 1000]
        bounds_y = [-np.inf, np.inf]
        
        _, _, parabola = generate_KB_parabola(f, L, R0, axis)
        parabola_intersection_f = generate_parabola_KB_intersection_function(f, L, R0, axis, z_low, z_up, bounds_x, bounds_y)
        (x_intersect, y_intersect, z_intersect), exists_intersection = parabola_intersection_f(x_0, e)
        np.testing.assert_almost_equal((x_intersect, y_intersect, z_intersect, exists_intersection), 
                                      (0.0, 50.0, 100.0, np.array(True)))
        



class TestExternalScript(unittest.TestCase):
    
    def test_run_absorption(self):
        result = subprocess.run(['python', 'examples/Plot_absorption.py'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)

    def test_run_WI(self):
        result = subprocess.run(['python', 'examples/try_Wolter_I.py'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)

    def test_run_Angel(self):
        result = subprocess.run(['python', 'examples/try_Lobster_Angel.py'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)

    def test_run_Schmidt(self):
        result = subprocess.run(['python', 'examples/try_Lobster_Schmidt.py'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)

    def test_run_KB(self):
        result = subprocess.run(['python', 'examples/try_KB.py'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
