import sys
import copy
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

from map_elites import MAPElites, Individual

from fitness_evaluator import Evaluator
from simulator import run_simulations_serial, run_simulations_parallel
from ndimarray import LabeledNDimArray, SequentialIterator  

from case_generator.combined_generator import CombinedGenerator
from case_generator.localization_generator import LocalizationGenerator
from case_generator.network_generator import NetworkGenerator
from case_generator.exploration_generator import ExplorationGenerator

from mapelites_train import Genome

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

from utilities import open_pickle

def main(visualize, parallel, logname):
    dims = [11,11,101]
    genome_mask = "11111111"

    repertoire = open_pickle(logname)

    solutions_with_indexes = repertoire.get_all_solutions()    
    solutions_without_indexes = map(lambda v: v[3], solutions_with_indexes)
    genomes = map(lambda v: v.o, solutions_without_indexes)

    new_repertoire = MAPElites(dims, None, None, 0) 
    new_repertoire._pop = new_repertoire._create_pop()

    def batch_evaluator(epoch, solutions):        
        case_configs = []

        case_configs.extend(CombinedGenerator([1000.0, 1000.0], 10, 10))

        eva = Evaluator(parametric=True)

        solutions_caseconfigs = []

        for si, solution in enumerate(solutions):
            #Push each simulation to a compute node
            # Inputs: Controller, Cases
            # Outputs: Fitness
            
            new_case_configs = []
            for case_config in case_configs:
                config_copy = copy.deepcopy(case_config)
                for platform_type in config_copy["platform_templates"].keys():
                    config_copy["platform_templates"][platform_type]["behavior"] = "MAPElitesParametric"
                    config_copy["platform_templates"][platform_type]["config_behavior"] = {}
                    config_copy["platform_templates"][platform_type]["config_behavior"]["interval"] = 0.5
                    config_copy["platform_templates"][platform_type]["config_behavior"]["mask"] = genome_mask
                    config_copy["platform_templates"][platform_type]["config_behavior"]["weights"] = solution._weights
                    config_copy["platform_templates"][platform_type]["config_behavior"]["center"] = solution._centers
                    config_copy["platform_templates"][platform_type]["config_behavior"]["spread"] = solution._spreads
                    config_copy["platform_templates"][platform_type]["config_behavior"]["scale"] = solution._scales

                config_copy['epoch'] = epoch
                config_copy['individual'] = si

                config_copy["config_simulator"] = {"view_delay": 6.0, "grid_size": [1000.0, 1000.0]}

                config_copy["config_simulator"]["max_time"] = 900.
                config_copy["config_simulator"]["log_delay"] = 200.0

                new_case_configs.append(config_copy)

            solutions_caseconfigs.append((solution, new_case_configs))

        if parallel:
            solution_logs = run_simulations_parallel(solutions_caseconfigs)#, False)
        else: 
            solution_logs = run_simulations_serial(solutions_caseconfigs, visualize)

        solutions_results = []
        for solution, logs in solution_logs:
            fitness, characteristics = eva.fitness_map_elites(logs)

            solutions_results.append((fitness, characteristics))
            
        return solutions_results
    
    batch_size = 200
    for i in range(len(genomes)/batch_size + 1):
        evaluation_results =  batch_evaluator(i, genomes[i*batch_size:min(len(genomes),(i+1)*batch_size)])

        for j, eval_result in enumerate(evaluation_results):
            individual = solutions_without_indexes[i*batch_size+j]
            fitness, characteristics = eval_result

            individual.characteristics = characteristics
            individual.fitness = fitness

            new_repertoire._place_solution(individual)

        print "Size of new repertoire", len(new_repertoire.get_all_solutions())

    new_repertoire.save_checkpoint("reeval.chkpt")

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parallel', dest='parallel', action='store_true')
    parser.set_defaults(parallel=False)
    parser.add_argument('--no_gui', dest='no_gui', action='store_true')
    parser.set_defaults(no_gui=False)

    parser.add_argument("logname",  type=str, help="Can be filename or folder")

    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    main(visualize=not args.no_gui, parallel=args.parallel, logname=args.logname)
