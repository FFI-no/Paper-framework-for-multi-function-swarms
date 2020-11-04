#!/usr/bin/python

from utilities import list_modules, load_module, select_class

class Agent(object):
    def __init__(self, config):        
        assert config.get("config_platform") is not None, "Configuration for platform was not found"
        assert config.get("type") is not None, "Type of platform was not specified"
        self.platform = self._setup_platform(config["type"],config["config_platform"])
        
        assert config.get("config_sensors") is not None, "Configuration for sensors was not found"
        self.sensors = self._setup_sensors(config["config_sensors"])
        
        if config.get("behavior") is not None:
            self.behavior = self._load_behavior(config["behavior"], config["config_behavior"])
        else:
            self.behavior = None
            print "Warning no behaviour initialized for %s" % (type(self.platform))
        
    def _setup_sensors(self, config_sensors):
        sensors = {}
        for sensor_name, sensor_parameters in config_sensors.items():
            module = load_module("sensors", sensor_name)
            sensor = getattr(module, sensor_name)
            sensors[sensor_name] = sensor(self, sensor_parameters)
            
        return sensors
            
    def _setup_platform(self, platform_type, config_platform):
        module = load_module("platforms", platform_type)
        
        platform = getattr(module, platform_type)
        
        return platform(self, config_platform)
        
    def _load_behavior(self, behavior, config_behavior):
        module = load_module("behaviors", behavior)
        
        behavior_class = getattr(module, behavior)
        
        return behavior_class(self, config_behavior)
    
    def bootstrap_events(self, current_time, case, *args, **kwargs):
        result_events = []
            
        events = self.platform.step(current_time, case, *args, **kwargs)
        result_events.extend(events)
        
        if self.behavior is not None:
            events = self.behavior.get_update(current_time, case)
            
            result_events.extend(events)
            
        return result_events
        
    def get_dict(self):
        return self.__getstate__()