import pickle
import numpy as np

from shapely.ops import cascaded_union
from shapely.geometry import Point, MultiPolygon

class Evaluator(object):
    def __init__(self, parametric=True):
        self._parametric=parametric
    
    def _exploration_fit(self, case, return_components = False):
        
        if case.agents[0].sensors.get("Coverage") is None:
            if return_components:
                return {}
            else:
                return 0.
        
        squares = []
        
        coverage = 0.
        for i in xrange(10):
            for j in xrange(10):                    
                squares.append(case.blackboard["Coverage"][i,j])

        squares = sorted(squares)
        num_agents = len(case.agents)
        max_velocity = case.agents[0].platform.max_velocity
        max_time = case.config["config_simulator"]["max_time"]
        max_squares = float(num_agents * max_velocity * max_time) / (100.)
        
        max_square_median = max_squares/ (10.*10.)

        median_frequency_normalized = float(squares[len(squares)/2]) / max_square_median

        if return_components:
            components = {}
            
            components["median"] = median_frequency_normalized
             
            return components
        else:
            fitness = median_frequency + coverage
            
            return fitness

    def _exploration_fit_live_delta(self, case1, case2, dt, return_components = False):
        if case2.agents[0].sensors.get("Coverage") is None:
            if return_components:
                return {}
            else:
                return 0.
            
        squares1 = []
        squares2 = []
        for i in xrange(10):
            for j in xrange(10):               
                if case1 is not None:
                    squares1.append(case1.blackboard["Coverage"][i,j])
                squares2.append(case2.blackboard["Coverage"][i,j])
                
        num_agents = len(case2.agents)
        max_velocity = case2.agents[0].platform.max_velocity
        max_squares = float(num_agents * max_velocity) / (100.)

        max_square_median = max_squares/ (10.*10.)

        median_frequency_normalized = float(squares2[len(squares2)/2]) / max_square_median

        coverage_delta = max(0.,sum(squares2)-sum(squares1))
        normalized_coverage_delta = (coverage_delta) / (max_squares * dt)


        if return_components:
            components = {}

            components["average"] = normalized_coverage_delta
             
            return components
        else:
            raise NotImplemented()
        
    def _localization_fit(self, case, return_components = False):
        if case.agents[0].sensors.get("Localization") is None:
            if return_components:
                return {}
            else:
                return 0.
            
        location_estimates = []
        
        for agent in case.agents:
            location_estimates.extend(agent.sensors.get("Localization").history)
            
        location_estimate_variance = 0.
        location_mean = np.array([0.,0.])
        
        actual_position = case.agents[0].sensors.get("Localization")._emitter_true_location
        
        for estimate in location_estimates:
            location_mean = location_mean + estimate

        location_mean = location_mean/len(location_estimates)
        
        for estimate in location_estimates:
            distance = np.linalg.norm(estimate-location_mean)
            
            location_estimate_variance += distance
            
        location_estimate_variance = location_estimate_variance/len(location_estimates)
        
        if return_components:
            components = {}
        
            components["variance"] = min(1.,location_estimate_variance/500.)
            
            return components
        else:
            fitness = min(1.,location_estimate_variance/500.)
            
            return fitness
        
    def _network_fit(self, case1, case2, return_components = False):
        if case1.agents[0].sensors.get("Relay") is None:
            if return_components:
                return {}
            else:
                return 0.

        covered_area_avg = 0.
        for case in [case1,case2]:
            
            connection_sets = []

            for agent in case.agents:
                member_of = []
                
                for i,connection_group in enumerate(connection_sets):
                    connected_to = False
                    
                    for connected_agent in agent.sensors.get("Relay").connections:
                        if connected_agent in connection_group:
                            connected_to = True
                    
                    if connected_to:
                        member_of.append((i,connection_group))
                
                if len(member_of) == 0:
                    new_set = set()
                    new_set.add(agent)
                    connection_sets.append(new_set)
                elif len(member_of) == 1:            
                    member_of[0][1].add(agent)
                else:
                    new_set = set()
                    
                    for i,connection_group in member_of:
                        new_set.update(connection_group)
                        
                    member_of.reverse()
                    for i,_ in member_of:
                        connection_sets.pop(i)
                        
                    connection_sets.append(new_set)
                    
            largest_set = max(connection_sets, key=lambda s: len(s))
            
            polygons = []

            com_range = 0.
            for agent in largest_set:
                x,y = agent.platform.position
                com_range = agent.sensors.get("Relay")._range
                p = Point(x,y).buffer(com_range)
                polygons.append(p)

            covered_area = cascaded_union(MultiPolygon(polygons)).area 
            max_covered_area = (len(case.agents)*3.14*com_range*com_range)/2.0
            covered_area_percentage = covered_area / max_covered_area

            covered_area_avg += covered_area_percentage

        if return_components:
            components = {}
            
            components["covered"] = min(1.,covered_area_avg/2.0)
            
            return components
        else:
            raise Exception("Not properly implemented")
            fitness =  float(len(largest_set))/len(case.agents)

            return fitness

    def _network_fit_live(self, case, return_components = False):
        if case.agents[0].sensors.get("Relay") is None:
            if return_components:
                return {}
            else:
                return 0.
            
        connection_sets = []

        for agent in case.agents:
            member_of = []
            
            for i,connection_group in enumerate(connection_sets):
                connected_to = False
                
                for connected_agent in agent.sensors.get("Relay").connections:
                    if connected_agent in connection_group:
                        connected_to = True
                
                if connected_to:
                    member_of.append((i,connection_group))
            
            if len(member_of) == 0:
                new_set = set()
                new_set.add(agent)
                connection_sets.append(new_set)
            elif len(member_of) == 1:            
                member_of[0][1].add(agent)
            else:
                new_set = set()
                
                for i,connection_group in member_of:
                    new_set.update(connection_group)
                    
                member_of.reverse()
                for i,_ in member_of:
                    connection_sets.pop(i)
                    
                connection_sets.append(new_set)
        
        largest_set = max(connection_sets, key=lambda s: len(s))
        
        polygons = []

        com_range = 0.
        for agent in largest_set:
            x,y = agent.platform.position
            com_range = agent.sensors.get("Relay")._range
            p = Point(x,y).buffer(com_range)
            polygons.append(p)


        covered_area = cascaded_union(MultiPolygon(polygons)).area 
        max_covered_area = (len(case.agents)*3.14*com_range*com_range)/2.0
        covered_area_percentage = covered_area / max_covered_area

        if return_components:
            components = {}
            
            components["covered"] = min(1.,covered_area_percentage)
            
            return components
        else:
            raise Exception("Not properly implemented")
            fitness =  float(len(largest_set))/len(case.agents)

            return fitness

    def _base_fit(self, case1, case2, return_components=False):  
        movement = 0.
        for agent_start, agent_end in zip(case1.agents, case2.agents):
            distance = np.linalg.norm(agent_start.platform.position-agent_end.platform.position)
            movement += min(1., distance/100.)
            
        if return_components:
            components = {}
            
            components["movement"] = movement/len(case1.agents)
            
            return components
        else:
                
            fitness = movement/len(case1.agents)
            
            return 1. + fitness
        
    def post_evaluator_loggers(self, list_of_loggers):
        summed = 1.
        
        for logger in list_of_loggers:
            with logger as open_logger:
                summed *= self._case_evaluator(open_logger._simulation_log)
            
        return summed
        
    def post_evaluator_shelves(self, list_of_shelves):
        summed = 1.
        
        for shelve_t in list_of_shelves:
            summed *= self._case_evaluator(shelve_t)
            
        return summed
        
    def _case_evaluator(self, run_log):
        ticks = sorted(map(int,run_log.keys()))
        
        start_case_id = str(min(ticks))
        end_case_id = str(max(ticks))
        
        start_case = run_log[start_case_id]
        end_case = run_log[end_case_id]
        
        
        fit_base = self._base_fit(start_case, end_case)
        fit_net = self._network_fit(end_case)
        fit_loc = self._localization_fit(end_case)
        fit_exp = self._exploration_fit(end_case)
        
        fitness = fit_base + fit_net + fit_loc + fit_exp
            
        return fitness
        
    def fitness_components(self, list_of_shelves):
        components = {}
        
        for shelve_t in list_of_shelves:
            components[case_name] = self._case_components(shelve_t)
        return components

    def _case_components(self, shelve):
        ticks = sorted(map(int,shelve.keys()))
        
        start_case_id = str(min(ticks))
        middle_case_id = str((max(ticks)-min(ticks))/2)
        end_case_id = str(max(ticks))
        
        start_case = shelve[start_case_id]
        middle_case = shelve[middle_case_id]
        end_case = shelve[end_case_id]
        
        case_name = str(start_case)
        case_component = {}
        
        case_component["network"] = self._network_fit(middle_case, end_case, return_components=True)
        case_component["localization"] = self._localization_fit(end_case, return_components=True)
        case_component["exploration"] = self._exploration_fit(end_case, return_components=True)
        
        return case_component


    def fitness_map_elites(self, list_of_loggers):
        test_fitness = 0.

        characteristics = {}
        
        for logger in list_of_loggers:
            with logger as open_logger:
                case_components = self._case_components(open_logger._simulation_log)
                
                for application, application_component in case_components.items():
                    for measure, value in application_component.items():

                        characteristics_name = "_".join([application,measure])
                        if characteristics.get(characteristics_name) is None:
                            characteristics[characteristics_name] = value
                        else:
                            characteristics[characteristics_name] += value

                config = open_logger._simulation_log['0'].config
                platform_name = config["platform_templates"].keys()[0]
                test_fitness += 2./(1.+np.linalg.norm(config["platform_templates"][platform_name]["config_behavior"]["weights"])+np.linalg.norm(config["platform_templates"][platform_name]["config_behavior"]["scale"]))
                
        for key,value in characteristics.items():
            characteristics[key] = value/len(list_of_loggers)

        test_fitness = test_fitness/len(list_of_loggers)

        return test_fitness, characteristics

    def fitness_from_simlog(self, simulation_log):
        characteristics = {}
        try:
            case_components = self._case_components(simulation_log)
        except:
            return None        
        for application, application_component in case_components.items():
            for measure, value in application_component.items():
                characteristics_name = "_".join([application,measure])
                characteristics[characteristics_name] = value

        config = simulation_log['0'].config
        platform_name = config["platform_templates"].keys()[0]
        test_fitness = 2./(1.+np.linalg.norm(config["platform_templates"][platform_name]["config_behavior"]["weights"])+np.linalg.norm(config["platform_templates"][platform_name]["config_behavior"]["scale"]))
      
        return test_fitness, characteristics

    def live_fitness(self, case1, case2, dt):
        characteristics = {}
                
        case_components = {}
        case_components["network"] = self._network_fit_live(case2, return_components=True)
        case_components["localization"] = self._localization_fit(case2, return_components=True)
        case_components["exploration"] = self._exploration_fit_live_delta(case1, case2, dt, return_components=True)

        for application, application_component in case_components.items():
            for measure, value in application_component.items():

                characteristics_name = "_".join([application,measure])
                if characteristics.get(characteristics_name) is None:
                    characteristics[characteristics_name] = value
                else:
                    characteristics[characteristics_name] += value

        config = case2.config
        platform_name = config["platform_templates"].keys()[0]
        test_fitness = 2./(1.+np.linalg.norm(config["platform_templates"][platform_name]["config_behavior"]["weights"])+np.linalg.norm(config["platform_templates"][platform_name]["config_behavior"]["scale"]))

        return test_fitness, characteristics


class LiveFitness(object):
    def __init__(self):
        self._evaluator = Evaluator()
        self._previous_case = None
        self._last_time = -1

    def evaluate(self, current_time, case):
        if self._previous_case is None:
            upcase = None
        else:
            upcase = pickle.loads(self._previous_case)
        r = self._evaluator.live_fitness(upcase, case, current_time-self._last_time)

        self._previous_case = pickle.dumps(case)
        self._last_time = current_time
        return r

if __name__=="__main__":
    import shelve, cPickle
    import argparse

    shelve.Pickler = cPickle.Pickler
    shelve.Unpickler = cPickle.Unpickler
                
    def create_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("filename", nargs=1, type=str)
        return parser

    parser = create_parser()
    args = parser.parse_args()

    eva = Evaluator()

    cases = shelve.open(args.filename[0])
    print   eva._case_components(cases)
    
