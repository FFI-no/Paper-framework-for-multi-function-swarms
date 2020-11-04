import shelve, cPickle
import argparse
import os
import shutil

from time import sleep
from visualize_case import VisualizeCase

import matplotlib.pyplot as plt

shelve.Pickler = cPickle.Pickler
shelve.Unpickler = cPickle.Unpickler

def main(logfilename, start_iteration = 0, show_figure=False, continous=False, delay=0, decimation=1, video_folder=None):
    cases = shelve.open(logfilename)
    
    viz  = VisualizeCase(cases[str(0)], show_figure=show_figure)

    viz.update(start_iteration, cases[str(start_iteration)] )
        
    if continous:
        for i in xrange(start_iteration+1, len(cases)):
            if i%decimation == 0:
                print "Iteration", i
                viz.update(i, cases[str(i)], folder=video_folder)
                
                if delay > 0:
                    sleep(delay)

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_iteration", nargs=1, type=int, default=[0])
    parser.add_argument("--delay", nargs=1, type=float, default=[0])

    parser.add_argument("--decimation", nargs=1, type=int, default=[1])

    parser.add_argument('--continous', dest='continous', action='store_true')
    parser.set_defaults(continous=False)
    parser.add_argument("filename", nargs=1, type=str)
    
    parser.add_argument("--video_folder", nargs=1, type=str, default=[None])


    parser.add_argument('--show_figure', dest='show_figure', action='store_true')
    parser.set_defaults(show_figure=False)
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    
    main(args.filename[0], args.start_iteration[0],show_figure=args.show_figure,continous=args.continous, delay=args.delay[0], decimation=args.decimation[0], video_folder=args.video_folder[0])
