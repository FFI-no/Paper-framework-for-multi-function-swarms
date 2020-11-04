import numpy as np


import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.patches as  mplpatch
    
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

from sensor import Sensor

from utilities import linalg_norm

#def vec_norm(vec):
#    return np.sqrt(np.dot(vec, vec)) 

class Relay(Sensor):
    def __init__(self, agent, config):
        super(Relay, self).__init__(agent, config)
        
        self.connections = []
        
        self._range = config["range"]
        
    def update(self, platform, case):
        super(Relay, self).update(platform, case)
        
        self.connections = []
        
        for agent in case.agents:
            if agent.platform is platform or not agent.platform.has_sensor(type(self)):
                continue
            
            #d = vec_norm(agent.platform.position - platform.position)
            d = linalg_norm(agent.platform.position - platform.position)
            if d < self._range:
                self.connections.append(agent)
                
    def get_patches(self):
        if self.position is not None:
            jet = cm = plt.get_cmap('Greens') 
            cNorm  = colors.Normalize(vmin=0, vmax=100)
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
            
            patches = []
            
            arrow  = mplpatch.ArrowStyle.Simple()
            for connection in self.connections:
                dx, dy = connection.platform.position - self.position
    
                c = (0,0,0)
                #Test that the arrow length is non-zero. Zero length error crashes matplotlib
                if dx**2 + dy**2 > 0.01:
                    patch = mplpatch.FancyArrow( self.position[0], self.position[1], dx, dy, head_width=0, width=4, zorder=-1, ec=c, fc=c )
                    patches.append(patch)
        
            return patches
        else:
            return []