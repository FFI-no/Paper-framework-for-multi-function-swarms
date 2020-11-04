import random
import numpy as np
import matplotlib.patches as  mplpatch

from platform import Platform

def vec_norm(vec):
    return np.sqrt(np.dot(vec, vec)) 

class MoveablePlatform(Platform):
    def __init__(self, agent, config):
        super(MoveablePlatform, self).__init__(agent, config)
        
        self.max_velocity = float(config["max_velocity"])
        self.max_acceleration = float(config["max_acceleration"])
        
        self.acceleration = np.array([0.0,0.0], float)
        self.velocity = np.array([0.0,0.0], float)
        self.position = np.array([random.randint(0,1000), random.randint(0,1000)], float)
        
        self.last_time = 0.
    
    def step(self, current_time, case):
        dt = current_time-self.last_time
        self.last_time = current_time

        self.velocity += self.acceleration*dt
        norm_velocity = vec_norm(self.velocity)
        if norm_velocity > self.max_velocity:
            self.velocity *= self.max_velocity/norm_velocity

        self.position += self.velocity*dt
        self.position = np.maximum(np.array([0.,0.]), np.minimum(np.array(case.config["config_simulator"]["grid_size"]), self.position))
        
        result_events = []
        events = super(MoveablePlatform, self).step(dt,case)

        result_events.extend(events)
        return result_events
        
    def set_position(self, position):
        raise NotImplemented
        
    def set_velocity(self, velocity):
        raise NotImplemented
        
    def set_acceleration(self, acceleration):
        raise NotImplemented
        
    def get_patches(self):
        patches = super(MoveablePlatform, self).get_patches()
        
        dx, dy = self.velocity
        patch = mplpatch.Arrow( self.position[0], self.position[1], dx, dy, 10, zorder=-2)
        
        patches.append(patch)
        
        return patches