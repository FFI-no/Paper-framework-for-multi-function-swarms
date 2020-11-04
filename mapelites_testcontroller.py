import random
import numpy as np
import argparse
import yaml
import matplotlib as plt
	
from case import Case
from logger import Logger
from agent import Agent
from visualize_case import VisualizeCase

from case_generator.combined_generator import CombinedGenerator
from case_generator.localization_generator import LocalizationGenerator
from case_generator.network_generator import NetworkGenerator
from case_generator.exploration_generator import ExplorationGenerator

import json

def load_case_config(filename):
	with open(filename) as handle:
		return yaml.load(handle)

def main(cont_type, cont_params, duration, n):    
	random.seed(0)

	print "Starting simulations"

	case_configs = []

	case_configs.extend(CombinedGenerator([1000.0, 1000.0], 1, 10))

	from fitness_evaluator import LiveFitness
	lf = LiveFitness()
	def fitness_evaluator(current_time, case):
		print lf.evaluate(current_time, case)

		return [(current_time+400., fitness_evaluator)]


	print cont_params
		
	for config in case_configs:
		for platform_type in config["platform_templates"].keys():
			cont_config = eval(cont_params)
			if cont_type=="weighted":
				config["platform_templates"][platform_type]["behavior"] = "MAPElitesWeighted"
				config["platform_templates"][platform_type]["config_behavior"]= {"interval": 0.5,"weights": cont_config}
			elif cont_type=="parametric":
				config["platform_templates"][platform_type]["behavior"] = "MAPElitesParametric"
				config["platform_templates"][platform_type]["config_behavior"] = {}
				config["platform_templates"][platform_type]["config_behavior"]["interval"] = 0.5
				config["platform_templates"][platform_type]["config_behavior"]["weights"] = cont_config["weights"]
				config["platform_templates"][platform_type]["config_behavior"]["center"] = cont_config["centers"]
				config["platform_templates"][platform_type]["config_behavior"]["spread"] = cont_config["spreads"]
				config["platform_templates"][platform_type]["config_behavior"]["scale"] = cont_config["scales"]
			else:
				raise Exception("Unknown controller type (%s)" % cont_type)

		if duration is None:
			config["config_simulator"]["max_time"] = 900.
		else:
			config["config_simulator"]["max_time"] = duration
			
		case = Case(config)
		visualization =VisualizeCase(case)	
		with Logger(case) as logger:
			with case as expanded_case:
				expanded_case.add_events([(0, fitness_evaluator)])
				expanded_case.run(logger=logger, visualization=visualization)
		
		visualization.close()
 
def create_parser():
	parser = argparse.ArgumentParser()

	parser.add_argument('--duration',type=int)

	parser.add_argument('--n',type=int, default=20)

	parser.add_argument('cont_type',type=str)
	parser.add_argument('cont_params',type=str)

	return parser

if __name__ == '__main__':
	parser = create_parser()
	args = parser.parse_args()
	
	main(args.cont_type, args.cont_params, args.duration, args.n)