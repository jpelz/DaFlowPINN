import yaml
import os
import itertools
import contextlib
import shutil
from datetime import datetime

class Config:
    """
    The Config class is responsible for managing the configuration settings for a Physics-Informed Neural Network (PINN) setup. 
    It initializes with default settings and can optionally load and merge configurations from a specified file. 
    The class provides methods to load, update, and unpack configuration dictionaries, as well as to run the PINN setup with the current configuration, 
    including support for parameter sweeps.

    Attributes:
        standard (dict): The default configuration settings.
        _conf_file (str): Path to the configuration file.
        _conf_dict (dict): The loaded configuration dictionary.

    Methods:
        __init__(self, conf_file: str = None): Initializes the Config object with default settings and optionally loads from a configuration file.
        _load_config(self, conf: str) -> dict: Loads the configuration file and merges it with the standard config.
        _update_config(self, base_config: dict, new_config: dict) -> dict: Updates the base configuration with loaded values.
        _unpack(self, d: dict, parent_key: str = ''): Recursively unpacks a nested dictionary into instance attributes.
        run(self, PINN_Setup: callable, print_to_log: bool = True): Runs the PINN setup with the current configuration, optionally performing parameter sweeps.
    """
    def __init__(self, conf_file: str = None):
        """
        Initialize the Config object with default settings and optionally load from a configuration file.

        :param conf_file: Path to the configuration file.
        """
        self.standard = {
            'architecture': {
                'model': 'fully_connected',
                'nr_layers': 5,
                'nr_hidden': 512,
                'hard_bc': False
            },
            'domain': {
                't_min': float('-inf'),
                't_max': float('inf')
            },
            'training': {
                'optim': {
                    'optimizer': 'adam',
                    'lr': 1e-3
                },
                'epochs': 1000,
                'print_freq': 10,
                'plot_freq': 100,
                'point_update_freq': None,
                'keep_percentage': 20,
                'amp_enabled': False
            },
            'validation': {
                'validate': False,
                'nt_max': 5,
                'nz_max': 5,
                'nx_max': 200,
                'ny_max': 100,
                'z_plot': 0,
                't_plot': [14, 14.5, 15]
            }
        }

        if conf_file is not None:
            self._conf_file = conf_file
            self._conf_dict = self.__load_config(self._conf_file)
            self._unpack(self._conf_dict)

    def __load_config(self, conf: str) -> dict:
        """
        Load configuration file and merge with standard config.

        :param conf: Path to the configuration file.
        :return: Merged configuration dictionary.
        """
        if not os.path.exists(conf):
            raise FileNotFoundError(f"Configuration file {conf} not found.")

        with open(conf, 'r') as file:
            user_config = yaml.safe_load(file)

        if 'base_config' in user_config and user_config['base_config'] is not None:
            if user_config['base_config'] == "standard":
                base_config = self.standard
            else:
                base_config = self.__load_config(user_config['base_config'])
            config = self.__update_config(base_config, user_config)
        else:
            config = user_config

        return config

    def __update_config(self, base_config: dict, new_config: dict) -> dict:
        """
        Update the base configuration with loaded values.

        :param base_config: The base configuration dictionary.
        :param new_config: The new configuration dictionary to merge.
        :return: Updated configuration dictionary.
        """
        for key, value in new_config.items():
            if isinstance(value, dict) and key in base_config:
                self.__update_config(base_config[key], value)
            else:
                base_config[key] = value

        return base_config

    def _unpack(self, d: dict, parent_key: str = ''):
        """
        Recursively unpack a nested dictionary into instance attributes.

        :param d: The dictionary to unpack.
        :param parent_key: The base key for nested attributes.
        """
        for key, value in d.items():
            new_key = f"{parent_key}{key}" if parent_key else key

            if isinstance(value, dict):
                nested_config = Config()
                nested_config._unpack(value)
                setattr(self, new_key, nested_config)
            else:
                setattr(self, new_key, value)

    def run(self, PINN_Setup: callable, print_to_log: bool = True, *args, **kwargs):
        """
        Executes the PINN (Physics-Informed Neural Network) setup, optionally performing a parameter sweep.

        This method either runs a single experiment or performs a sweep over specified configuration parameters.
        If parameter sweeps are defined (as lists in the configuration), it iterates over all combinations,
        creating a separate directory and log file for each run. Otherwise, it runs a single experiment.

        Args:
            PINN_Setup (callable): The function to execute for each experiment, which takes the configuration as its first argument.
            print_to_log (bool, optional): If True, redirects stdout to a log file for each run. Defaults to True.
            *args: Additional positional arguments to pass to PINN_Setup.
            **kwargs: Additional keyword arguments to pass to PINN_Setup.

        Notes:
            - Parameter sweeps are detected as list attributes in the configuration (excluding those containing 'validation' in their name).
            - Each run is executed in its own directory, named after the experiment and sweep index.
            - Log files are created with timestamps if print_to_log is True.
            - The working directory is restored after each run.
        """
        def find_sweep_params(obj, prefix=""):
            sweep_params = {}
            for key, value in obj.__dict__.items():
                if isinstance(value, Config):
                    sub_params = find_sweep_params(value, prefix=f"{prefix}{key}.")
                    sweep_params.update(sub_params)
                elif isinstance(value, list):
                    sweep_params[f"{prefix}{key}"] = value
            return sweep_params

        sweep_params = find_sweep_params(self)
        filtered_sweep_params = {k: v for k, v in sweep_params.items() if 'validation' not in k}
        sweep_params = filtered_sweep_params

        headdir = os.getcwd()
        if not sweep_params:
            if not os.path.exists(f'{headdir}/{self.name}'):
                os.makedirs(f'{headdir}/{self.name}')
            os.chdir(f'{headdir}/{self.name}')
            if print_to_log:
                logfile = f'{self.name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
                with open(logfile, 'w') as log:
                    with contextlib.redirect_stdout(log):
                        print(f"Running {self.name}:")
                        PINN_Setup(self, *args, **kwargs)
            else:
                PINN_Setup(self, *args, **kwargs)
            os.chdir(headdir)
        else:
            sweepdir = f'{headdir}/{self.name}'
            if not os.path.exists(sweepdir):
                os.makedirs(sweepdir)
            param_combinations = list(itertools.product(*sweep_params.values()))
            for j, combination in enumerate(param_combinations):
                current_conf = Config(self._conf_file)
                current_conf.name = f"{self.name}_{j}"
                if not os.path.exists(f'{sweepdir}/{current_conf.name}'):
                    os.makedirs(f'{sweepdir}/{current_conf.name}')
                os.chdir(f'{sweepdir}/{current_conf.name}')
                if print_to_log:
                    logfile = f'{current_conf.name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
                    with open(logfile, 'w') as log:
                        with contextlib.redirect_stdout(log):
                            for i, param_name in enumerate(sweep_params.keys()):
                                keys = param_name.split(".")
                                target_obj = current_conf
                                for key in keys[:-1]:
                                    target_obj = getattr(target_obj, key)
                                setattr(target_obj, keys[-1], combination[i])
                            print(f"Running {current_conf.name} with parameters:")
                            for k, key in enumerate(sweep_params):
                                print(f"  {key}: {combination[k]}")
                            PINN_Setup(current_conf, *args, **kwargs)
                os.chdir(headdir)
            

        

#Alternative decorator for running the PINN setup with a configuration file

def main(conf_file: str, print_to_log: bool = True):
    """
    Decorator to run a function with a configuration file.
    
    Use:
        @main(conf_file='path/to/config.yaml')
        def PINN_Setup(conf, *args, **kwargs):
            # Your PINN setup code here
            ...
    """

    def decorator(PINN_Setup):
        def wrapper(*args, **kwargs):
            conf = Config(conf_file)

            def find_sweep_params(obj, prefix=""):
                sweep_params = {}
                for key, value in obj.__dict__.items():
                    if isinstance(value, Config):
                        sub_params = find_sweep_params(value, prefix=f"{prefix}{key}.")
                        sweep_params.update(sub_params)
                    elif isinstance(value, list):
                        sweep_params[f"{prefix}{key}"] = value
                return sweep_params

            sweep_params = find_sweep_params(conf)
            filtered_sweep_params = {k: v for k, v in sweep_params.items() if 'validation' not in k}
            sweep_params = filtered_sweep_params

            headdir = os.getcwd()
            if not sweep_params:
                if not os.path.exists(f'{headdir}/{conf.name}'):
                    os.makedirs(f'{headdir}/{conf.name}')
                os.chdir(f'{headdir}/{conf.name}')
                shutil.copyfile(conf._conf_file, f'{conf.name}.yaml')
                if print_to_log:
                    logfile = f'{conf.name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
                    with open(logfile, 'w') as log:
                        with contextlib.redirect_stdout(log):
                            print(f"Running {conf.name}:")
                            PINN_Setup(conf, *args, **kwargs)
                else:
                    PINN_Setup(conf)
                os.chdir(headdir)
            else:
                sweepdir = f'{headdir}/{conf.name}'
                if not os.path.exists(sweepdir):
                    os.makedirs(sweepdir)
                shutil.copyfile(conf._conf_file, f'{sweepdir}/{conf.name}.yaml')
                param_combinations = list(itertools.product(*sweep_params.values()))
                for j, combination in enumerate(param_combinations):
                    current_conf = Config(conf._conf_file)
                    current_conf.name = f"{conf.name}_{j}"
                    if not os.path.exists(f'{sweepdir}/{current_conf.name}'):
                        os.makedirs(f'{sweepdir}/{current_conf.name}')
                    os.chdir(f'{sweepdir}/{current_conf.name}')
                    if print_to_log:
                        logfile = f'{current_conf.name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
                        with open(logfile, 'w') as log:
                            with contextlib.redirect_stdout(log):
                                for i, param_name in enumerate(sweep_params.keys()):
                                    keys = param_name.split(".")
                                    target_obj = current_conf
                                    for key in keys[:-1]:
                                        target_obj = getattr(target_obj, key)
                                    setattr(target_obj, keys[-1], combination[i])
                                print(f"Running {current_conf.name} with parameters:")
                                for k, key in enumerate(sweep_params):
                                    print(f"  {key}: {combination[k]}")
                                PINN_Setup(current_conf, *args, **kwargs)
                    os.chdir(headdir)
        return wrapper
    return decorator

    
