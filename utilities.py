import os
import sys
import pickle
import shelve
import traceback

from math import sqrt

def linalg_norm(vector):
    return sqrt(vector[0]**2 + vector[1]**2)

def open_pickle(filename):
    if filename is None or not os.path.exists(filename):
        return None

    f = open(filename,"r")
    data = pickle.load(f)
    f.close()

    return data

def open_shelve(filename):
    if filename is None or not os.path.exists(filename):
        return None

    return shelve.open(filename)

def list_modules(folder):
    types = []
    for f in os.listdir(folder):
        if os.path.isfile(os.path.join(folder,f)):
            try:
                name, extension = f.split(".")
                if extension == "py" and name != "__init__":
                    types.append(name)
            except:
                continue
    return types
    
def load_module(folder,name):
    try:
        module_list = list_modules(folder)
        module_list = dict([(filename.replace("_",""), filename) for filename in module_list]) 
        
        module = __import__("%s.%s" % (folder,module_list[name.lower()]), fromlist=[name])
    except ImportError:
        # Display error message
        traceback.print_exc(file=sys.stdout)
        raise ImportError("Failed to import module {0} from folder {1}".format(name,folder))
    return module

def select_class(name, name_list):
    for n, c in name_list:
        if n == name:
            return c

class Translator(object):
    def __init__(self, pairs):
        self._lookup = dict(pairs)

    def __getitem__(self, value):
        if self._lookup.get(value) is not None:
            return self._lookup.get(value)
        else:
            return value