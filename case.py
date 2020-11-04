import yaml
import Queue
import random

import numpy as np

from agent import Agent
from utilities import linalg_norm

class Blackboard(object):
    def __init__(self):
        self.objects = {}
        
    def get_object(self, name, generator=None):
        if self.objects.get(name) is not None:
            return self.objects.get(name)
        else:
            if generator is None:
                return
            else:
                self.objects[name] = generator()
                return self.objects[name]
                
    def __getitem__(self, name):
        return self.get_object(name)

    def reset_object(self, name):
        data_type = self.objects[name].__class__
        del self.objects[name]
        self.objects[name] = data_type()

class Case(object):
    def __init__(self, config):
        self.config = config
        
        self.config["seed"] = random.random()
        
        self._expanded = False
        self._halted = False
        
        self._previous_positions = None
        
    def __str__(self):
        case_name = ""
        if self.config.get('epoch') is not None:
            case_name += "Epoch %s/" % self.config.get('epoch')

        if self.config.get('generation') is not None:
            case_name += "Generation %s/" % self.config.get('generation')        
                
        if self.config.get('individual') is not None:
            case_name += "I-%s " % self.config.get('individual')
        
        case_name += self.config['name'] 
        
        return case_name

    def _init_agents(self, config_agents, platform_templates):
        agents = []
        for agent_type, agent_count in config_agents.items():
            
            assert platform_templates.get(agent_type) is not None, "No suitable template could be found for %s" % agent_type
            
            for i in range(agent_count):
                agents.append(Agent(platform_templates[agent_type]))
        
        return agents
        
    def __enter__(self):
        if self._expanded:
            raise Exception("Case already expanded")
        
        random.seed(self.config["seed"])
        
        assert self.config.get("config_agents") is not None, "Configuration for agents was not found"
        assert self.config.get("platform_templates") is not None, "Platform templates for agents was not found"
        
        self.agents = self._init_agents(self.config["config_agents"], self.config["platform_templates"])
        
        self._expanded = True
        self._halted = False
        
        self.priority_queue = Queue.PriorityQueue()
        
        self.blackboard = Blackboard()

        self.current_time = 0.
        
        return self
        
    def __exit__(self, type, value, traceback):
		self._expanded = False
		
		del self.blackboard
		del self.priority_queue
		for agent in self.agents:
			del agent
		
		return

    def add_events(self, events):
        for event in events:
            self.priority_queue.put(event)
        
    def _init_events(self, logger, visualization):
        for agent in self.agents:
            for event_time, event in agent.bootstrap_events(0, self):
                self.priority_queue.put((event_time, event))
        
        if logger is not None:
            logger.set_log_delay(self.config["config_simulator"]["log_delay"])
            logger_event = logger.update(0, self)
            self.priority_queue.put(logger_event[0])
        
        if visualization is not None:
            visualization_event = visualization.update(0, self)
            self.priority_queue.put(visualization_event[0])
            
        check_stalled_event = self._check_stalled_simulation(0.0, self)
        self.priority_queue.put(check_stalled_event[0])
            
    def _check_stalled_simulation(self, current_time, case):
        new_positions = [agent.platform.position for agent in self.agents]
        
        if self._previous_positions is not None:
            distance_moved = 0.
            
            for old_position, new_position in zip(self._previous_positions, new_positions):
                delta = linalg_norm(old_position-new_position)
                
                distance_moved += delta
            
            if distance_moved < 0.1:
                self._halted = True
                print "Simulated stalled, exiting"
                
        self._previous_positions = new_positions
        
        return [(current_time + 10., self._check_stalled_simulation)]

    def run(self, logger, visualization=None):
        if not self._expanded:
            raise Exception("Case not expanded")
        
        assert self.config.get("config_simulator") is not None, "Configuration for simulator was not found"
        
        #Initalize variables
        random.seed(self.config["seed"])
        max_time = self.config["config_simulator"]["max_time"]
        
        #Generate initial events for time 0.
        self._init_events(logger, visualization)
        
        #Process events while simulation durations limit is not reached and no other reason to halt has been found
        while self.current_time < max_time and not self._halted:
            #Get a new event to process
            self.current_time, event = self.priority_queue.get()
                
            #Process event (this can be anything from agent tick to logging or visalization
            event_results = event(self.current_time, self)
            
            #Put new resulting events into the queue
            [self.priority_queue.put(new_event) for new_event in event_results]

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        if state.get('priority_queue') is not None:
            del state['priority_queue']
        if state.get('_previous_positions') is not None:
            del state['_previous_positions']
        return state
        
    def get_dict(self):
        return self.__getstate__()