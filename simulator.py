import tasks
import time

from time import sleep
from case import Case
from logger import Logger
from visualize_case import VisualizeCase   

from redis.exceptions import ConnectionError     

class TaskState(object):
    def __init__(self, delayed_callable, meta_data=[]):
        self._callable = delayed_callable
        self._delayed_task = None
        self._meta_data = meta_data

        self._start_time = -1

        self._args = None
        self._kwargs = None

    def issue(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._delayed_task = self._callable(*self._args, **self._kwargs)
        self._start_time = time.time()

    def reissue(self):
        self._delayed_task.revoke(terminate=True)
        self._delayed_task = self._callable(*self._args, **self._kwargs)
        self._start_time = time.time()

    def revoke(self):
        self._delayed_task.revoke(terminate=True)

    def status(self):
        count = 3
        e = None
        for i in range(3):
            try: 
                return self._delayed_task.status
            except ConnectionError as e:
                count += 1

        print "Failed getting status"
        raise e

    def has_timedout(self):
        if self.status() == "SUCCESS":
            return False
        delta = time.time() - self._start_time
        return delta > 60.*5

class TaskGroup(object):
    def __init__(self, delayed_callable):
        self._callable = delayed_callable

        self._tasks = []

        self._retried = 0
        self._failed = 0
        self._started = 0
        self._timedout = 0

    def add_task(self, meta_data, *args, **kwargs):
        async_task = TaskState(self._callable, meta_data)
        async_task.issue(*args, **kwargs)
        self._tasks.append(async_task)
        self._started += 1

    def _filter_statuses(self, statuses):
        pending = len(filter(lambda x: x == "PENDING", statuses))
        finished = len(filter(lambda x: x == "SUCCESS", statuses))
        failed =  len(filter(lambda x: x == "FAILURE", statuses))
        
        return {"started": self._started, "timedout": self._timedout, "pending": pending, "finished": finished, "failed": self._failed+failed, "retried": self._retried }

    def get_statuses(self):
        statuses = []

        for async_task in self._tasks:
            statuses.append(async_task.status())

        filtered_statuses = self._filter_statuses(statuses)
        return filtered_statuses

    def _check_on_tasks(self):
        statuses = []

        for i,task_state in enumerate(self._tasks):
            status = task_state.status()
            if task_state.has_timedout() or status == "FAILURE":
                if status == "FAILURE":
                    self._failed += 1
                else:
                    self._timedout += 1

                task_state.reissue()
                status = task_state.status()
                self._retried += 1
            statuses.append(status)

        filtered_statuses = self._filter_statuses(statuses)
        return filtered_statuses

    def run(self):
        statuses = self.get_statuses()

        task_count = len(self._tasks)
        finished_count = statuses["finished"]

        try:
            while statuses["finished"] < task_count:
                statuses = self._check_on_tasks()
                sleep(1.)

                if statuses["finished"] != finished_count:
                    self.print_statuses(statuses)

                finished_count = statuses["finished"]

        except (AssertionError, KeyboardInterrupt, SystemExit), e:
            for async_task in self._tasks:
                async_task.revoke()
            print "Exiting."
            raise SystemExit            

    def print_statuses(self, statuses=[]):
        if len(statuses) == 0:
            statuses = self.get_statuses()

        for key, value in statuses.items():
            print key, value

        print

    def results(self):
        statuses = self.get_statuses()
        if statuses["finished"] < len(self._tasks):
            print "Not finished, wait a bit before requesting results"
            return None

        results =  [(async_task._meta_data, async_task._delayed_task.result) for async_task in self._tasks]
        [async_task._delayed_task.forget() for async_task in self._tasks]

        return results

def run_simulations_parallel(pairs):
    tg = TaskGroup(tasks.run_case.delay)

    solutions = {}
    for i, (solution, case_configs) in enumerate(pairs):
        solutions[i] = solution
        for case_config in case_configs:
            tg.add_task(i, case_config)

    tg.run()

    results_per_solution = {}
    for i, log in tg.results():
        if results_per_solution.get(i) is None:
            results_per_solution[i] = []

        results_per_solution[i].append(log)

    return [(solution, results_per_solution[i]) for i, solution in solutions.items()]

def run_simulation(config, visualize=False):
    case = Case(config)

    if visualize:
        visualization = VisualizeCase(case)
    else:
        visualization = None
    
    logger = Logger(case, None, prefix_folder="logs")
    with logger as open_logger:

        with case as expanded_case:
            expanded_case.run(logger=open_logger, visualization=visualization)
        
    print "Completed simulation: %s" % str(case)

    if visualization is not None:
        visualization.close()
    
    return logger

def run_simulations_serial(pairs, visualize=False):
    k_logs = []

    for k, case_configs in pairs:
        logs  = [run_simulation(case_config, visualize) for case_config in case_configs]  
        k_logs.append((k, logs))

    return k_logs