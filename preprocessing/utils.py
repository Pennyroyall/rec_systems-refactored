import io
import six
from time import time
import numpy as np
import pickle
import scipy.sparse as sparse
import logging
import sys
import random
import os


from datetime import timedelta
from datetime import datetime
from collections import defaultdict
from gensim.models import Word2Vec

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 

path_data = './../data/'
path_log = './../logs/'
path_res = './../results/'



def pickle_dump(data, filename):
    with io.open(filename, "wb") as f:
        pickle.dump(data, f)

    
def pickle_load(filename):
    with io.open(filename, "rb") as f:
        return pickle.load(f)


def timeit(message=None):
    def decor(method):
        def timed(*args, **kw):
            ts = time()
            result = method(*args, **kw)
            te = time()
            if message:
                to_print = f'{message} {te-ts:.1f} seconds\n'
            else:
                to_print = f'{method.__name__.upper()} is done in {te-ts:.1f} seconds \n'
            print(to_print)
            return result
        return timed
    return decor            
