import numpy as np
import matplotlib.patches as mplpatch

from sensor import Sensor

class Optical(Sensor):
    def __init__(self, config, platform):
        super(Optical, self).__init__(config, platform)
        
        self.location_history = []
        
        self._fov = config["fov"]
        self._altitude = config["altitude"]
    
    def get_patches(self):
        if self.position is not None:
            
            width_coverage = np.arctan(self._fov/360.0 * 3.14)*2.0 * self._altitude 
            
            ground_coverage = np.array([1, 0.75])* width_coverage
            
            patches = []
            
            patch = mplpatch.Rectangle(
                    self.position-ground_coverage/2, 
                    ground_coverage[0],        
                    ground_coverage[1],        
                    0,
                    alpha=0.3,
                    zorder=-1
                )
                
            patches.append(patch)

            return patches
            
        else:
            return []