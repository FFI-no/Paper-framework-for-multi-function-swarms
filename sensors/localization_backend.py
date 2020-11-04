import random
import numpy as np

from deap import base
from deap import creator
from deap import tools

emitter_strength = 100.0
loss_factor_a = 2.0
noise_stddev = 2.0

def squared_norm(vec):
    return np.dot(vec,vec)

class LocalizationBackend(object):
    def __init__(self):
        self._emitter_strength = emitter_strength
        self._emitter_position = None
        self._loss_factor_a = loss_factor_a
        self._noise_stddev = noise_stddev

    def set_emitter_position(self, emitter_position):
        self._emitter_position = emitter_position

    def _emitterStrengthDbm(self):
        return 10.0*np.log10(self._emitter_strength)
        
    def _actualPower(self, pos):
        n_pos = np.array(pos)
        d = np.sqrt(squared_norm(n_pos-self._emitter_position))
        if d == 0: 
            return self._emitterStrengthDbm()
        else:
            return self._emitterStrengthDbm() - 10.0 * self._loss_factor_a * np.log10(d)
        
    def measuredPower(self, pos):
        a = self._actualPower(pos)
        n = random.gauss(0,self._noise_stddev)
        
        return (pos,  a+n)

    def _pkl(self, samples, k, l):
        return samples[k][1]- samples[l][1]
    
    def _qxy(self, samples, position):
        error = 0.0
        for k in xrange(len(samples)):
            for l in xrange(k+1,len(samples)):
                p = self._pkl(samples,k,l) - 5.0 * loss_factor_a * np.log10(squared_norm(position-samples[l][0])/squared_norm(position-samples[k][0]))
                error += p**2
        return error

class BruteforceLocalization(LocalizationBackend):
    def predict_location(self, samples, grid_size, step_size):
        grid_steps = np.array(grid_size / step_size, int)
        
        best_loc = None
        best_value = None
        for i in xrange(grid_steps[0]):
            for j in xrange(grid_steps[1]):
                loc = np.array([i,j], float)*step_size
                value = self._qxy(samples, loc)
                
                if best_loc is None or best_value > value:
                    best_value = value
                    best_loc = loc
        
        
        return best_loc

class NMLocalization(LocalizationBackend):
    def __init__(self, *args, **kwargs):
        super(NMLocalization, self).__init__(*args, **kwargs)

        creator.create("FitnessMinError", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMinError)
            
        self._toolbox = base.Toolbox()    
        self._toolbox.register("select", tools.selBest)

        def _generateIndividual():
            ind = creator.Individual(random.uniform(0,1000.0) for i in range(2)) 
            return ind
            
        self._toolbox.register("individual", _generateIndividual)

        self._toolbox.register("population", tools.initRepeat, list, self._toolbox.individual)

    def predict_location(self, samples, grid_size, step_size):
        alpha = 1.0
        gamma = 2.0
        phi = -0.5
        sigma = 0.5
        
        nm_evaluations = 40
        max_evaluations = 100

        #Select 3 random samples from the available samples, to reduce the computation required for an estimation
        selected_samples = random.sample(samples, 3)
        
        def fitness(ind):
            return self._qxy(selected_samples, ind),
        
        self._toolbox.register("evaluate", fitness)
        
        #Create a population of random individuals
        pop = self._toolbox.population(n=max_evaluations-nm_evaluations)
        
        # Evaluate the individuals
        fitnesses = self._toolbox.map(self._toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        #THIS SHOULD NOT BE len(pop), NM wont run. Set to len(pop) to be compatible with results previously generated.
        evaluations = len(pop)
        #evaluations = 0

        pop = tools.selBest(pop, 3)
        
        while evaluations < nm_evaluations:
            l_sorted = sorted(pop, key=lambda i: i.fitness.values[0])
            
            best = l_sorted[0]
            worst = l_sorted[-1]
            
            centroid = reduce(lambda x, y: x+y, map(lambda i: np.array(i), l_sorted[:-1]), np.zeros(2))/float(len(pop)-1)
            
            reflected_point = centroid + alpha * (centroid - np.array(worst))
            reflected_fitness = self._toolbox.evaluate(reflected_point)
            evaluations += 1
            
            if reflected_fitness < best.fitness.values[0]:
                expansion_point = centroid + gamma * (centroid - np.array(worst))
                expansion_fitness = self._toolbox.evaluate(expansion_point)
                
                pop = list(l_sorted[:-1])
                if expansion_fitness[0] < reflected_fitness[0]:
                    ind = creator.Individual(expansion_point)
                    ind.fitness.values = self._toolbox.evaluate(ind)
                    evaluations += 1
                    pop.append(ind)
                else:
                    ind = creator.Individual(reflected_point)
                    ind.fitness.values = self._toolbox.evaluate(ind)
                    evaluations += 1
                    pop.append(ind)
                
            else:
                contraction_point = centroid + phi * (centroid - np.array(worst))
                contraction_fitness = self._toolbox.evaluate(contraction_point)
            
                if contraction_fitness[0] < worst.fitness.values[0]:
                    pop = list(l_sorted[:-1])
                    
                    ind = creator.Individual(contraction_point)
                    ind.fitness.values = self._toolbox.evaluate(ind)
                    evaluations += 1
                    pop.append(ind)
                else:
                    new_pop = list()
                    
                    for i in range(1, len(pop)):
                        xi = np.array(best) + sigma * (np.array(l_sorted[i] - np.array(best)))

                        ind = creator.Individual(xi)
                        ind.fitness.values = self._toolbox.evaluate(ind)
                        evaluations += 1
                        new_pop.append(ind)
                        
                    new_pop.append(best)
                    pop = new_pop
                
        best = tools.selBest(pop, 1)[0]
        
        return np.array(best)