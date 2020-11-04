import random
import numpy as np
import matplotlib.patches as mplpatch
import matplotlib as mpl

from sensor import Sensor

class CoverageMap(object):
    def __init__(self):
        self.history = np.zeros((10,10))
        
    def get_square(self, i, j):
        return self.history[i][j]
        
    def incr_square(self, i,j):
        self.history[i][j] += 1
        
    def get_patches(self):
        patches = []
        
        max_value = 15
        print self.history.max()
        min_value = 0
        
        for i, row in enumerate(self.history):
            for j, value in enumerate(row):
                position = np.array([i,j])* 100.0
                if max_value - min_value > 0:
                    alpha = float(value- min_value)/float(max_value - min_value) 
                else:
                    alpha = 0.

                patch = mplpatch.Rectangle(position, 100,  100, 0, alpha=min(1.,max(0.,alpha)), zorder=-100)
            
                patches.append(patch)
                
        return patches
        
    def __getitem__(self, (i, j)):
        return self.get_square(i,j)

class Coverage(Sensor):
    def __init__(self, agent, config):
        super(Coverage, self).__init__(agent, config)
        
        self._new_pos = True
        self.old_i,self.old_j = -1, -1
        
    def update(self, platform, case):
        i,j = map(int,platform.position/100.0)
        
        i = min(9,max(0,i))
        j = min(9,max(0,j))
        
        if i != self.old_i or j != self.old_j:
            self.old_i,self.old_j = i,j
            self._new_pos = True
            
            case.blackboard.get_object(self.__class__.__name__, CoverageMap).incr_square(i,j)