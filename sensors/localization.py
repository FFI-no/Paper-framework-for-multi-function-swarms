import numpy as np
import collections
import matplotlib.patches as mplpatch

from sensor import Sensor
from localization_backend import *

backend = NMLocalization()

class Localization(Sensor):
    def __init__(self, agent, config):
        super(Localization, self).__init__(agent, config)
        
        self.localization = None

        self._backend = backend
        self._backend.set_emitter_position(config["emitter_position"])

        #Not known by the agents only used for visualization
        self._emitter_true_location = config["emitter_position"]
        
        self.history = []
        
        self.state = 0
        
    def get_sample(self):
        return self._backend.measuredPower(self.position)
    
    def update(self, platform, case):
        super(Localization, self).update(platform, case)

        if self.state == 0:
        
            self.velocity = platform.velocity
            
            samples = [agent.sensors[self.__class__.__name__].get_sample() for agent in filter(lambda agent: agent.platform.has_sensor(type(self)), case.agents)]
            
            self.localization = self._backend.predict_location(samples, np.array([1000, 1000], float), 10.0)
            
            self.history.append(self.localization)
            
            if len(self.history) > 20:
                self.history.pop(0)
        
        self.state = (self.state+1)%5

    def get_patches(self):
        patches = []

        patch = mplpatch.Circle(
                    self._emitter_true_location,   
                    8,  
                    zorder=-2,
                    color="y",
                    fill=True)
        patches.append(patch)

        patch = mplpatch.Circle(
                self.localization,  
                3,  
                zorder=-2,
                color=(0.,0.,0.),
                fill=True)
        patches.append(patch)
        
        for position in self.history:
            patch = mplpatch.Circle(
                position,   # (x,y)
                4,  
                zorder=-2,
                color=(0.4,0.4,0.4),
                fill=True)
                
            patches.append(patch)                
        
        return patches

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        if state.get("_backend") is not None:
            del state['_backend']
        return state