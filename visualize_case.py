import time 
import random
import os.path
import logging

import numpy as np
import pylab as p
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mpl_toolkits.mplot3d.axes3d as p3

class Timer(object):
    def __enter__(self):
        self.start_time = time.clock()
        return self
        
    def __exit__(self, type, value, traceback):
        self.end_time = time.clock()
    
    def lapsed(self):
        return self.end_time - self.start_time
        
class VisualizeCase(object):
    def __init__(self, case, disable_axis=False, show_figure=True):
        self.figure_id = str(case)
        self.figure = plt.figure(self.figure_id, figsize=(7,7))
        self.ax = self.figure.gca()
        self.ax.set_aspect("equal")
        
        self.grid_size = np.array(case.config["config_simulator"]["grid_size"])
        
        plt.xlim([0,self.grid_size[0]])
        plt.ylim([0,self.grid_size[1]])
        
        if disable_axis:
            plt.axis("off")
        else:
            plt.xlabel('X')
            plt.ylabel('Y')
        
        self.patch_handles = []
        
        self.view_delay = case.config["config_simulator"]["view_delay"]
        
        self.events = []
        
        FORMAT = '%(asctime)-15s  %(message)s'
        logging.basicConfig(format=FORMAT)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.WARNING)

        self._show_figure = show_figure
        if self._show_figure:
            p.show(block=False)
    
    def _select_self(self):
        p.figure(self.figure_id)
    
    def _update_agents(self, agents):
        for patch in self.patch_handles:
            patch.remove()
        
        self._select_self()
            
        agent_positions = []
        
        self.patch_handles = []
        
        for agent in agents:
            agent_positions.append(agent.platform.position)
            
            for sensor in agent.sensors.values():
                patches = sensor.get_patches()
                self.patch_handles.extend(patches)
                
            patches = agent.platform.get_patches()
            self.patch_handles.extend(patches)
            
            patches = agent.behavior.get_patches()
            self.patch_handles.extend(patches)
        
        
        for patch in self.patch_handles:
            self.ax.add_artist(patch)
            
            
        xv = map(lambda x: x[0], agent_positions)
        yv = map(lambda x: x[1], agent_positions)
        patch = p.scatter(xv,yv, c='r',s=30)
        self.patch_handles.append(patch)
        
    def _update_blackboard(self, blackboard):
    
        patches = []
        for data in blackboard.objects.values():
            patches.extend(data.get_patches())
        
        self.patch_handles.extend(patches)
        for patch in patches:
            self.ax.add_artist(patch)
        
        
    def update(self, current_time, case, folder=None):
        t = Timer()
        with t:
            self._update_agents(case.agents)
            self._update_blackboard(case.blackboard)
            
            self.figure.canvas.draw()
            
            if folder is not None:
                self.figure.savefig(os.path.join(folder, "frame-%06d.png" % (int(current_time))), transparent=True)
       
        self.logger.info( "Plotting took %s seconds", t.lapsed())
        
        event = (current_time + self.view_delay, self.update)
        return [event]
        
    def get_events(self):
        r = self.events
        
        self.events = []
        
        return r
    
    def close(self):
        self._select_self()
        plt.close()