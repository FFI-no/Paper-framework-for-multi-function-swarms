import random
import numpy as np
import matplotlib as mpl
import matplotlib.patches as  mplpatch

from map_elites_base import MAPElitesBase
from math import atan, cos, sin, exp
from utilities import linalg_norm

class MAPElitesParametric(MAPElitesBase):
    def __init__(self, agent, config):
        super(MAPElitesParametric, self).__init__(agent,config)
        
        self.set_parameters(self.config["weights"], self.config["center"], self.config["spread"], self.config["scale"])

        self._savedpatches = []

    def set_parameters(self, weights, centers, spreads, scales):
        assert len(weights)==8, "Wrong number of weight parameters "
        assert len(centers)==8, "Wrong number of center parameters "
        assert len(spreads)==8, "Wrong number of spread parameters "
        assert len(scales)==8, "Wrong number of scale parameters "

        self._weights = weights
        center_spreads = zip(centers[:6], spreads[:6])
        center_spreads = sorted(center_spreads)
        self._center = map(lambda v: v[0], center_spreads)
        self._center.extend(centers[6:])
        self._spread = map(lambda v: v[1], center_spreads)
        self._spread.extend(spreads[6:])
        self._scale = scales

    def _calc_force_contribution(self, directions, ranges):
        assert len(directions)==8, "Expected 8 directions got %s" % len(directions)
        assert len(ranges)==8, "Expected 8 ranges got %s" % len(ranges)

        total_force = np.array([0.,0.])

        self._forces = []
        for direction, range_, weight, center, spread, scale in zip(directions, ranges, self._weights, self._center, self._spread, self._scale):
            if direction is None:
                continue

            direction -= 3.14/2.
            direction = float(int(direction*100)%628)/100.

            unit_direction = np.array([np.cos(direction), np.sin(direction)])

            if range_ is None:
                if weight < 0.01:
                    continue
                force = unit_direction*weight
            else:                
                exponent = -(range_-center)/max(0.00001,spread)
                exponent = min(8.,max(-8.,exponent))
                sigmoid = ((2./(1. + exp(exponent))) - 1.) * weight

                exponent = -(range_-center)**2/max(0.00001,spread)**2
                exponent = min(8.,max(-8.,exponent))
                well = -2*(range_-center) * scale * exp(exponent)
                
                force = unit_direction*(sigmoid+well)

            total_force = total_force + force
            self._forces.append(force)

        return total_force/len(directions)

    def get_update(self, current_time, case):
        if self.agent.platform.position is None:
            print "Warning position is none"
            return 

        directions, ranges = self._generate_inputs(case, self.agent.platform.position)
        velocity = self._calc_force_contribution(directions, ranges)

        self.agent.platform.set_velocity(velocity*self.agent.platform.max_velocity)

        dt = self.config["interval"]
        return [(current_time+dt, self.get_update)]
        
    def get_patches(self):
        patches = self._savedpatches[:]

        return patches

        for force in self._forces:
            force = force * 10.
            force_norm = linalg_norm(force)

            patch = mplpatch.Arrow( self.agent.platform.position[0], self.agent.platform.position[1], force[0], force[1], 10, zorder=-2)
            patches.append(patch)

        return patches