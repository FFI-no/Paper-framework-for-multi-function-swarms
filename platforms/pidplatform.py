import numpy as np

from moveable_platform import MoveablePlatform

def vec_norm(vec):
    return np.sqrt(np.dot(vec, vec)) 

class PIDPlatform(MoveablePlatform):
    def __init__(self, agent, config):
        super(PIDPlatform, self).__init__(agent, config)
        
        self.setpoint_position = None
        self.setpoint_velocity = None
        
    def set_position(self, position):
        position = np.array(position, float)
        
        self.setpoint_position = position
        
    def set_velocity(self, velocity):
        velocity = np.array(velocity, float)

        self.setpoint_position = None
        self.setpoint_velocity = velocity
        
    def set_acceleration(self, acceleration):
        acceleration = np.array(acceleration, float)
        
        self.setpoint_position = None
        self.setpoint_velocity = None
        
        norm_acceleration = vec_norm(acceleration) 
        if norm_acceleration > self.max_acceleration:
            acceleration *= self.max_acceleration/norm_acceleration
        
        self.acceleration = acceleration
        
    def _controller_update(self):
        if self.setpoint_position is not None:
            delta = self.setpoint_position - self.position
            delta_length = vec_norm(delta)
            
            if delta_length > self.max_velocity:
                delta *= self.max_velocity/delta_length
                
            self.setpoint_velocity = delta
        
        if self.setpoint_velocity is not None:
            delta = self.setpoint_velocity - self.velocity
            delta_length = vec_norm(delta)
            
            if delta_length > self.max_acceleration:
                delta *= self.max_acceleration/delta_length
                
            self.acceleration = delta
