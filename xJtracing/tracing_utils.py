import numpy as np
import yaml

def assert_single_number(*args):
    for arg in args:
        assert isinstance(arg, (int, float)), print('wrong array shape', arg)
        
def assert_array_1d(*args):
    for arg in args:
        assert len(arg.shape)==1, print('wrong array shape', arg)

def assert_array_2d(*args):
    for arg in args:
        assert len(arg.shape)==2, print('wrong array shape', arg)

def assert_array_3d(*args):
    for arg in args:
        assert len(arg.shape)==3, print('wrong array shape', arg)
        

def save_telescope_pars(telescope_pars, savepath):
    def numpy_to_list(data):
        if isinstance(data, np.ndarray):
            return data.tolist()  # Convert numpy array to list
        elif isinstance(data, dict):
            return {key: numpy_to_list(value) for key, value in data.items()}  # Recursively handle dict
        elif isinstance(data, list):
            return [numpy_to_list(item) for item in data]  # Handle list
        return data
    with open(savepath, 'w') as file:
        yaml.dump(numpy_to_list(telescope_pars), file, default_flow_style=False)

def load_telescope_pars(savepath):
    def list_to_numpy(data):
        if isinstance(data, list):
            return np.array(data)  # Convert list back to numpy array
        elif isinstance(data, dict):
            return {key: list_to_numpy(value) for key, value in data.items()}  # Recursively handle dict
        elif isinstance(data, list):
            return [list_to_numpy(item) for item in data]  # Handle list
        return data
    with open(savepath, 'r') as file:
        telescope_pars = yaml.load(file, Loader=yaml.FullLoader)
    return list_to_numpy(telescope_pars)