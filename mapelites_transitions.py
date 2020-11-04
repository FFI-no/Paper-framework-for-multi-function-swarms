#!/usr/bin/python2.7
import random
import numpy as np
import argparse
import yaml

from case import Case
from logger import Logger
from agent import Agent

from case_generator.combined_generator import CombinedGenerator
from case_generator.localization_generator import LocalizationGenerator
from case_generator.network_generator import NetworkGenerator
from case_generator.exploration_generator import ExplorationGenerator

try:
    import tasks

    from simulator import wait_for_parallel_completion
except:
    print "No parallell support"
    
def load_case_config(filename):
    with open(filename) as handle:
        return yaml.load(handle)

from fitness_evaluator import LiveFitness

import matplotlib.pyplot as plt
class TransitionManager(object):
	def __init__(self, behavior_list, behavior_interval, tick_interval):
		self._lf_evaluator = LiveFitness()
		self._i = 0

		self._behavior_list = behavior_list
		self._behavior_interval = behavior_interval
		self._tick_interval = tick_interval

		self._fig = plt.figure("characteristics")
		plt.ion()
		plt.show()

		self._series = {}

	def handle_change_behavior(self, current_time, case):

		print self._behavior_list[self._i]
		self._i = (self._i+1)%len(self._behavior_list)

		for agent in case.agents:
			agent.behavior.set_parameters(self._behavior_list[self._i]["weights"], self._behavior_list[self._i]["centers"], self._behavior_list[self._i]["spreads"], self._behavior_list[self._i]["scales"])

		case.blackboard.reset_object("Coverage")

		return [(current_time+self._behavior_interval, self.handle_change_behavior)]

	def _add_to_series_and_make_if_none(self, name, value):

		if self._series.get(name) is None:
			self._series[name] = []
		self._series[name].append(value)

	def handle_live_fitness(self, current_time, case):
		fitness, characteristics = self._lf_evaluator.evaluate(current_time, case)
		self._add_to_series_and_make_if_none("fitness", fitness)
		for key, value in characteristics.items():
			self._add_to_series_and_make_if_none(key , value)

			if len(self._series[key][-5:]) > 0:
				value = sum(self._series[key][-5:])/float(len(self._series[key][-5:]))
			else:
				value = 0.
			self._add_to_series_and_make_if_none(key + "_avg" , value)

		return [(current_time+self._tick_interval, self.handle_live_fitness)]

	def handle_plot(self, current_time, case):
		plt.figure("characteristics")
		plt.clf()

		for name, series in self._series.items():
			if name[-3:] == "avg":
				plt.plot(range(len(series)), series,label= name)

		plt.legend()
		self._fig.canvas.draw()

		plt.show()

		return [(current_time+200.,self.handle_plot)]

	def finished(self):
		plt.figure("characteristics")
		plt.close()


behavior_list = [
{"weights": [0,0,0,0,0,0,0,100], "centers": np.ones(8)*100., "spreads": np.ones(8)*300., "scales": np.array([-0.2,-0.1,-0.1,-0.1,-0.1,-0.1,0.,0.])/2.},
{"weights": [-10,0,0,0,0,0,0,0], "centers": np.ones(8), "spreads": np.ones(8), "scales": np.zeros(8)},
{"weights": [0,0,0,0,0,-1,0,0], "centers": np.ones(8)*200., "spreads": np.ones(8)*300., "scales": np.array([-0.2,-0.1,-0.1,-0.1,-0.1,-0.1,0.,0.])/2.},
{"weights": [5,0,0,0,0,0,10,0], "centers": np.ones(8), "spreads": np.ones(8), "scales": np.zeros(8)}
]

class DataAggregator(object):
	def __init__(self, behavior_list, num_trials, max_time, fitness_interval):
		self._behavior_list = behavior_list
		self._max_time = max_time
		self._fitness_interval = fitness_interval

		self._trials = []

		self._current_trman = None

	def init_events(self, expanded_case):
		if self._current_trman is not None:
			self.finished()

		self._current_trman = TransitionManager(self._behavior_list, self._max_time, self._fitness_interval)

		expanded_case.add_events([(0,self._current_trman.handle_change_behavior)])
		#expanded_case.add_events([(0,self._current_trman.handle_plot)])
		#expanded_case.add_events([(0,self._current_trman.handle_live_fitness)])

	def plot(self):
		if self._current_trman is not None:
			self.finished()

		series = self._aggregate_series()

		plt.figure("characteristics")
		plt.ioff()

		for name, series in series.items():
			plt.plot(range(len(series)), series,label= name)

		plt.legend()

		plt.show()

	def _aggregate_series(self):
		agg_series = {}
		for trial in self._trials:
			for name, values in trial._series.items():
				if agg_series.get(name) is None:
					agg_series[name] = values
				else:
					agg_series[name] = np.array(agg_series[name]) + np.array(values)

		return agg_series

	def finished(self):
		self._trials.append(self._current_trman)
		self._current_trman = None

def main(visualize=True,  parallel=True):    
	random.seed(0)

	print "Starting simulations"

	num_trials = 20
	max_time_per_behavior = 900. 
	max_time = max_time_per_behavior*len(behavior_list)
	fitness_interval = 20.
	da = DataAggregator(behavior_list, num_trials, max_time_per_behavior, fitness_interval)

	case_configs = CombinedGenerator([1000.0, 1000.0], num_trials, 5)

	results_inprogress = []

	for config in case_configs:
		config["config_simulator"]["max_time"] = max_time

		print config

		if parallel:
			#Run externally
			r = tasks.run_case.delay(config)
			results_inprogress.append(r)
					
		else:
			#Run locally
			case = Case(config)
			if visualize:
				from visualize_case import VisualizeCase
				visualization = VisualizeCase(case)
			else:
				visualization = None
				
			with Logger(case) as logger:
				with case as expanded_case:
					da.init_events(expanded_case)
					expanded_case.run(logger=logger, visualization=visualization)
			
			if visualize:
				visualization.close()
	da.plot()

	if parallel:
		wait_for_parallel_completion(results_inprogress)

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parallel', dest='parallel', action='store_true')
    parser.set_defaults(parallel=False)
    parser.add_argument('--no_gui', dest='no_gui', action='store_true')
    parser.set_defaults(no_gui=False)
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    
    
    main(visualize=not args.no_gui, parallel=args.parallel)
    
