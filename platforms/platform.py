import random
import numpy as np
import matplotlib.patches as  mplpatch

id = 0

class Platform(object):
    def __init__(self, agent, config):
        self.agent = agent
        
        self.position = np.array([random.randint(0,1000), random.randint(0,1000)], float)

        global id

        self._id = id
        id += 1
        
    def has_sensor(self, sensor_type, string=False):
        if string:
            for sensor in self.agent.sensors.values():
                if sensor_type in str(type(sensor)):
                    return True
                    
            return False

        else:
            for sensor in self.agent.sensors.values():
                if type(sensor) == sensor_type:
                    return True
                    
            return False
        
    def step(self, current_time, case):
        for sensor in self.agent.sensors.values():
            sensor.update(self, case)

        return []
        
    def set_position(self, position):
        raise NotImplemented
        
    def get_patches(self):
        return  []
        
        patch = mplpatch.Circle(
                self.position,
                10,  
                zorder=0,
                color="r")
        return [patch]


    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        if state.get("agent") is not None:
            del state['agent']  
        return state
        
    def get_dict(self):
        return self.__getstate__()
