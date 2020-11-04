import numpy as np
import random

class BaseCase(object):
    def __init__(self, *args):
        self._dict = {}

        self._setup(*args)

    def _setup(self, index, grid_size, name, num_agents):
        self["name"] = name
        self["config_agents"] = {"a": num_agents}
        self["platform_templates"] = {"a": {}}

        self["platform_templates"]["a"]["type"] = "Quadcopter"
        self["platform_templates"]["a"]["config_platform"] = {"max_velocity": 10.0, "max_acceleration": 1.0, "interval": 1.0}
        self["platform_templates"]["a"]["config_sensors"] = {}
        
        self["config_simulator"] = {"max_time": 1000.0, "view_delay": 6.0, "log_delay": 30.0, "grid_size": grid_size}

    def __getitem__(self, key):
        return self._dict.__getitem__(key)

    def __setitem__(self, key, value):
        self._dict.__setitem__(key,value)

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def get(self, key):
        return self._dict.get(key)

    def __repr__(self):
        return self._dict.__repr__()

class ConfigGenerator(object):  
    def __init__(self, grid_size, num_cases, num_agents):
        self._grid_size = np.array(grid_size)
        self._num_cases = num_cases

        assert num_agents > 3, "Minimum number of agents is 4"
        self._num_agents = num_agents

        self._current_index = 0

        self._base_case = BaseCase
            
    def __iter__(self):
        np.random.seed(random.randint(0,1000000))
        
        self._current_index = 0
        
        return self
        
    def next(self):
        if self._current_index >= self._num_cases:
            raise StopIteration
            
        name = "%s %s" % (self.__class__.__name__, self._current_index)
        
        config = self._base_case(self._current_index, self._grid_size, name, self._num_agents)
        
        self._current_index += 1
        
        return config