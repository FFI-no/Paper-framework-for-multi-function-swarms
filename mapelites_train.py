import sys
import copy
import random
import argparse
import numpy as np

from map_elites import MAPElites
from fitness_evaluator import Evaluator
from simulator import run_simulations_serial, run_simulations_parallel

from case_generator.combined_generator import CombinedGenerator
from case_generator.localization_generator import LocalizationGenerator
from case_generator.network_generator import NetworkGenerator
from case_generator.exploration_generator import ExplorationGenerator

class Genome(object):
    def __init__(self, size, genome_mask):
        self._size = size
        self._genome_mask = genome_mask
        #init random

        self._weight_range = 5.
        self._weights = np.random.rand(size)*self._weight_range*2-np.ones(size)*self._weight_range
        self._centers = np.random.rand(size)*1000.
        self._spreads = np.random.rand(size)*100.
        self._scales = np.random.rand(size)*1.-np.ones(size)*0.5

        for i in range(size):
            if not genome_mask[i]:
                self._weights[i] = 0.
                self._scales[i] = 0.

    def clone(self):
        i = Genome()

        i._weights = self._weights
        i._centers = self._centers
        i._spread = self._spreads
        i._scale = self._scales

    def mutate(self):
        i = random.randint(0, self._size*4-1)

        li = i%self._size

        #Pick a new random index as long as one of the masked input vectors are chosen
        while not self._genome_mask[li]:
            i = random.randint(0, self._size*4-1)
            li = i%self._size

        if i < self._size:   
            #mutate weights
            self._weights[li] += random.gauss(0., self._weight_range/5.)
            self._weights[li] = max(-self._weight_range, min(self._weight_range, self._weights[li])) 

        elif self._size < i < self._size*2:
            #mutate centers
            self._centers[li] += random.gauss(0., 100.)
            self._centers[li] = max(0., min(1000., self._centers[li])) 

        elif 3*self._size < i < 4*self._size:
            #mutate spread
            self._spreads[li] += random.gauss(0., 10.)
            self._spreads[li] = max(0., min(100., self._spreads[li])) 

        else:
            #mutate scale
            self._scales[li] += random.gauss(0., 0.1)
            self._scales[li] = max(-0.5, min(0.5, self._scales[li])) 


    def dict(self):
        return {"weights": list(self._weights), "centers": list(self._centers), "spreads": list(self._spreads), "scales": list(self._scales)}
            
    def __str__(self):
        import json

        t = {"weights": list(self._weights), "centers": list(self._centers), "spreads": list(self._spreads), "scales": list(self._scales)}
        return json.dumps(t)

def main(visualize, parallel, genome_mask, num_evals, num_epochs, test_mode=False):

    print "Genome mask:", genome_mask
    print "Num evals:", num_evals
    print "Num epochs:", num_epochs

    import numpy as np
    import matplotlib.pyplot as plt

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import ListedColormap

    if test_mode:
        print >>sys.stderr, "*****TEST MODE ENABLED*****"
    
    if test_mode:
        dims = [11,11,101]
    else:
        dims = [11,11,101]#,11,11]#,11,11]

    solution_size = 4

    gi = 0

    def batch_evaluator(epoch, solutions):        
        case_configs = []

        if test_mode:
            case_configs.extend(CombinedGenerator([1000.0, 1000.0], num_evals, 10))
        else:
            case_configs.extend(CombinedGenerator([1000.0, 1000.0], num_evals, 10))

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

        import shutil
        shutil.rmtree("logs")
            
        return solutions_results

    def mutate(genome):
        genome.mutate()
        return genome
    
    def generate():
        return Genome(8,genome_mask)

    if test_mode:
        m = MAPElites(dims, generate, mutate, 10, batch_evaluator=batch_evaluator) 
    else:
        m = MAPElites(dims, generate, mutate, 200, batch_evaluator=batch_evaluator) 
    m.init()

    if test_mode:
        m.run_batch(num_epochs+1,10)
    else:
        m.run_batch(num_epochs+1,200)

def parse_genome_mask(mask):
    assert(len(mask) == 8)
    return map(int, mask)

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parallel', dest='parallel', action='store_true')
    parser.set_defaults(parallel=False)
    parser.add_argument('--no_gui', dest='no_gui', action='store_true')
    parser.set_defaults(no_gui=False)
    parser.add_argument('--test_mode', dest='test_mode', action='store_true')
    parser.set_defaults(test_mode=False)

    genome_mask = "11111111"# [0,1,0,1,1,1,1,1]
    parser.add_argument('--genome_mask', dest='genome_mask')
    parser.set_defaults(genome_mask=genome_mask)

    parser.add_argument('--num_evals', dest='num_evals')
    parser.set_defaults(num_evals=5)

    parser.add_argument('--num_epochs', dest='num_epochs')
    parser.set_defaults(num_epochs=200)

    parser.add_argument('--seed', dest='seed')
    parser.set_defaults(seed=0)

    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    random.seed(args.seed)

    main(visualize=not args.no_gui, parallel=args.parallel, genome_mask=parse_genome_mask(args.genome_mask), num_evals=int(args.num_evals), num_epochs=int(args.num_epochs), test_mode=args.test_mode)