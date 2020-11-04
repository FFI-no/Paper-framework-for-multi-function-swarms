import numpy as np

from config_generator import BaseCase, ConfigGenerator

class CombinedCase(BaseCase):
    def __init__(self, *args, **kwargs):
        super(CombinedCase, self).__init__(*args, **kwargs)

    def _setup(self, index, *args, **kwargs):
        super(CombinedCase, self)._setup(index, *args, **kwargs)

        self["platform_templates"]["a"]["behavior"] = "MAPElitesParametric"
        self["platform_templates"]["a"]["config_behavior"] = {}
        self["platform_templates"]["a"]["config_behavior"]["interval"] = 1.
        self["platform_templates"]["a"]["config_behavior"]["weights"] = [0,0,0,0,1,1,0,5]
        self["platform_templates"]["a"]["config_behavior"]["center"] = [150.,150.,150.,150.,150.,150.,1000.,0.]
        self["platform_templates"]["a"]["config_behavior"]["spread"] = [60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0]
        self["platform_templates"]["a"]["config_behavior"]["scale"] =  [-0.9,-0.1,-0.1,-0.1,-0.1,-0.1,0,0]

        self["platform_templates"]["a"]["config_sensors"]["Coverage"] = {}
        self["platform_templates"]["a"]["config_sensors"]["Relay"] = {"range": 200}
        emitter_positions = [np.array([200.,200.]), np.array([200.,800.]), np.array([800.,800.]), np.array([800.,200.]), np.array([500.,500.])]
        self["platform_templates"]["a"]["config_sensors"]["Localization"] = {"emitter_position": emitter_positions[index%len(emitter_positions)]}

class CombinedGenerator(ConfigGenerator):
    def __init__(self, grid_size, num_cases, num_agents):
        super(CombinedGenerator, self).__init__(grid_size, num_cases, num_agents)

        self._base_case = CombinedCase