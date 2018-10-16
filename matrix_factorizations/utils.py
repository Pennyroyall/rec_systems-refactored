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


from implicit.als import AlternatingLeastSquares
from numba import jit
from numba import jitclass
from numba import int32, float32


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


def dcg_score(vector):
    # return = np.sum(vector / np.log2(np.arange(len(vector)) + 2))
    return vector[0] + np.sum(vector[1:] / np.log2(np.arange(2, vector.size + 1)))


def matrix_ndcg(P, Q, R_train, R_test, n=10, p=1):
    U, I = np.shape(R_train)
    corr = 0
    ndcg = 0
    COUNT = 0
    for u in range(U):
        if np.random.rand()<=p:
            COUNT += 1
            temp = np.asarray(P[u, :].dot(Q)).flatten()
            indices_train = R_train[u, :].indices
            temp[indices_train] = -np.inf
            indices_top = np.argpartition(temp, -n)[-n:]
            indices_pred = np.argsort(temp[indices_top])[::-1]
            pred = indices_top[indices_pred]

            l = min(n, len(R_test[u, :].indices))
            vector = np.zeros(n)
            for i in range(n):
                if R_test[u, pred[i]] > 0:
                    vector[i] = 1

            if l>0:
                score = dcg_score(vector)
                ideal = dcg_score(np.ones(l))
                ndcg += score/ideal
            else:
                corr += 1
    return ndcg / (COUNT)