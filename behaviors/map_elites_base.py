import random
import numpy as np

from behavior import Behavior
from math import atan, cos, sin, exp, sqrt

from utilities import linalg_norm

class MAPElitesBase(Behavior):
    def __init__(self, agent, config):
        self.agent = agent
        self.config = config     

        #Used to calculate average of PDOA estimates over time
        self.localization_history = []

        self._least_visited_angle = None
        
    def _generate_inputs(self, case, self_position):          
        ranges = []
        directions = []
                
        neighbours = []
        
        for agent in case.agents:
            if self.agent == agent:
                continue 
                
            position = agent.platform.position
            delta = position-self_position
            #distance = np.linalg.norm(delta)
            distance = linalg_norm(delta)
            direction = 3.14-np.arctan2(*delta)
            neighbours.append((direction, distance))
            
        neighbours = sorted(neighbours, key=lambda v: v[1])
        
        for i in range(6):
            if len(neighbours) <= i:
                directions.append(None)
                ranges.append(None)
            else:
                direction, distance = neighbours[i]

                directions.append(direction)
                ranges.append(distance)

        if self.agent.sensors.get('Coverage') is not None:
            if self.agent.sensors.get('Coverage')._new_pos:
                x,y = map(int,self_position/100.0)

                squares = []
                for o_i in range(-1, 2, 1):
                    for o_j in range(-1, 2, 1):
                        if o_i == 0 and o_j == 0:
                            continue

                        if 0 <= o_i + x <= 9 and 0 <= o_j + y <= 9:
                            squares.append(((o_i+x, o_j+y), case.blackboard["Coverage"][x+o_i,y+o_j]))
                            
                _, min_value = min(squares, key=lambda v: v[1])
                equal_to_min_squares = filter(lambda v: v[1] <= min_value+0.01, squares)
                min_direction, min_value = random.choice(equal_to_min_squares)

                delta = np.array(min_direction)*100. + np.array([50.,50.]) - self_position
                angle = 3.14-np.arctan2(*delta)
            

                self._least_visited_angle = angle
                self.agent.sensors.get('Coverage')._new_pos = False


            directions.append(self._least_visited_angle)
            ranges.append(None)

        else:
            directions.append(None)
            ranges.append(None)
            
        if self.agent.sensors.get('Localization') is not None:
            self.localization_history.append(self.agent.sensors['Localization'].localization)

            avg_position = reduce(lambda x,y: x+y, self.localization_history) / len(self.localization_history)
            
            delta = avg_position - self_position
            distance = linalg_norm(delta)
            direction = 3.14-np.arctan2(*delta)
            
            directions.append(direction)
            ranges.append(distance)
        else:
            directions.append(None)
            ranges.append(None)
                    
        return directions, ranges