#!/usr/bin/python

import sys, os, pickle
sys.path
sys.path.append(os.getcwd())

from celery import Celery
from billiard import current_process

from case import Case
from logger import Logger

app = Celery("tasks")
app.config_from_object('celeryconfig')

@app.task
def run_case(config):
    case = Case(config)
    try:
        process_id = current_process().index
    except:
        process_id = 0
    
    logger = Logger(case, None, prefix_folder=os.path.join(".tmp",str(process_id)))
    with logger as open_logger:
        with case as expanded_case:
            expanded_case.run(logger=open_logger)
    
    print "Completed simulation: %s" % str(case)

    return logger