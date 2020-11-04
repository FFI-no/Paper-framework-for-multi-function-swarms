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

behavior_list = [{"spreads": [16.82697964512554, 77.14201287718751, 57.5259176527892, 92.47819610008156, 44.94761979357678, 17.517320590632757, 19.073021929279008, 22.58486720820868], "weights": [-1.0412719913752722, -1.6678700083822386, -0.26032460804698865, 1.6336062760239987, -2.4984455496272844, -1.960932704368748, 0.0468216013851015, 0.8623250639185704], "centers": [315.87028042906195, 0.0, 956.3180303842147, 997.3296848075914, 1.9627262554283709, 414.4359095325473, 78.75330054161833, 26.24035086372856], "scales": [0.3250242358166051, -0.12273495195084683, -0.25442546686137274, -0.06851856230041986, 0.08584172427862635, 0.08851917495286554, -0.19916561006040945, 0.14303659398806778]}]

class TransitionManager(object):
	def __init__(self):
		self._lf_evaluator = LiveFitness()
		self._i = 0

	def handle_event(self, current_time, case):
		print self._lf_evaluator.evaluate(current_time, case)

		print behavior_list[self._i]
		self._i = (self._i+1)%len(behavior_list)

		for agent in case.agents:
			agent.behavior.set_parameters(behavior_list[self._i]["weights"], behavior_list[self._i]["centers"], behavior_list[self._i]["spreads"], behavior_list[self._i]["scales"])

		return [(current_time+200., self.handle_event)]

def main(visualize=True,  parallel=True):    
	random.seed(0)

	print "Starting simulations"

	case_configs = []

	case_configs.extend(CombinedGenerator([1000.0, 1000.0], 1, 9))

	trman = TransitionManager()

	results_inprogress = []

	for config in case_configs:
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
					expanded_case.add_events([(0,trman.handle_event)])
					expanded_case.run(logger=logger, visualization=visualization)
			
			if visualize:
				visualization.close()
		
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