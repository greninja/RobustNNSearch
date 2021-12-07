import math
import numpy as np
import multiprocessing
from pathlib import Path
import sys
import random
from collections import defaultdict
from sklearn.cluster import KMeans

default_params = \
    {
        # min/max of parameters
        "min": -1.0,
        "max": 1.0,
        # only useful if you use the 'iso_dd' variation operator
        "iso_sigma": 0.01,
        "line_sigma": 0.2,
    }

class Species:
    def __init__(self, x, desc, fitness, centroid=None):
        self.x = x
        self.desc = desc
        self.fitness = fitness
        self.centroid = centroid

# def polynomial_mutation(x):
#     '''
#     Cf Deb 2001, p 124 ; param: eta_m
#     '''
#     y = x.copy()
#     eta_m = 5.0
#     r = np.random.random(size=len(x))
#     for i in range(0, len(x)):
#         if r[i] < 0.5:
#             delta_i = math.pow(2.0 * r[i], 1.0 / (eta_m + 1.0)) - 1.0
#         else:
#             delta_i = 1 - math.pow(2.0 * (1.0 - r[i]), 1.0 / (eta_m + 1.0))
#         y[i] += delta_i
#     return y

def sbx(x, y, params):
    '''
    SBX (cf Deb 2001, p 113) Simulated Binary Crossover

    A large value ef eta gives a higher probablitity for
    creating a `near-parent' solutions and a small value allows
    distant solutions to be selected as offspring.
    '''
    eta = 10.0
    xl = params['min']
    xu = params['max']
    z = x.copy()
    r1 = np.random.random(size=len(x))
    r2 = np.random.random(size=len(x))

    for i in range(0, len(x)):
        if abs(x[i] - y[i]) > 1e-15:
            x1 = min(x[i], y[i])
            x2 = max(x[i], y[i])

            beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
            alpha = 2.0 - beta ** -(eta + 1)
            rand = r1[i]
            if rand <= 1.0 / alpha:
                beta_q = (rand * alpha) ** (1.0 / (eta + 1))
            else:
                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

            c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

            beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
            alpha = 2.0 - beta ** -(eta + 1)
            if rand <= 1.0 / alpha:
                beta_q = (rand * alpha) ** (1.0 / (eta + 1))
            else:
                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
            c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

            c1 = min(max(c1, xl), xu)
            c2 = min(max(c2, xl), xu)

            if r2[i] <= 0.5:
                z[i] = c2
            else:
                z[i] = c1
    return z

def iso_dd(x, y, params):
    '''
    Iso+Line
    Ref:
    Vassiliades V, Mouret JB. Discovering the elite hypervolume by leveraging interspecies correlation.
    GECCO 2018
    '''
    assert(x.shape == y.shape)
    p_max = np.array(params["max"])
    p_min = np.array(params["min"])
    a = np.random.normal(0, params['iso_sigma'], size=len(x))
    b = np.random.normal(0, params['line_sigma'])
    norm = np.linalg.norm(x - y)
    z = x.copy() + a + b * (x - y)
    return np.clip(z, p_min, p_max)

def variation(x, y, variation_type, params):
    assert(x.shape == y.shape)
    if variation_type == "sbx":
        z = sbx(x, y, params)
    elif variation_type == "iso_dd":
        z = iso_dd(x, y, params)
    return z

def __centroids_filename(k, dim):
    return 'centroids_' + str(k) + '_' + str(dim) + '.dat'

def __write_centroids(centroids):
    k = centroids.shape[0]
    dim = centroids.shape[1]
    filename = __centroids_filename(k, dim)
    with open(filename, 'w') as f:
        for p in centroids:
            for item in p:
                f.write(str(item) + ' ')
            f.write('\n')

def cvt(k, dim, samples, cvt_use_cache=True):
    # check if we have cached values
    fname = __centroids_filename(k, dim)
    if cvt_use_cache:
        if Path(fname).is_file():
            print("WARNING: using cached CVT:", fname)
            return np.loadtxt(fname)
            
    # otherwise, compute cvt
    print("Computing CVT (this can take a while...):", fname)

    x = np.random.rand(samples, dim)
    # k_means = KMeans(init='k-means++', n_clusters=k,
    #                  n_init=1, n_jobs=-1, verbose=1)#,algorithm="full")
    k_means = KMeans(init='k-means++', n_clusters=k,
                     n_init=1, verbose=1) #,algorithm="full")
    k_means.fit(x)
    __write_centroids(k_means.cluster_centers_)

    return k_means.cluster_centers_


def make_hashable(array):   
    return tuple(map(float, array))


def parallel_eval(evaluate_function, to_evaluate, pool, params):
    if params['parallel'] == True:
        s_list = pool.map(evaluate_function, to_evaluate)
    else:
        s_list = map(evaluate_function, to_evaluate)
    return list(s_list)

# format: fitness, centroid, desc, genome \n
# fitness, centroid, desc and x are vectors
def __save_archive(archive, gen):
    def write_array(a, f):
        for i in a:
            f.write(str(i) + ' ')
    filename = 'archive_' + str(gen) + '.dat'
    with open(filename, 'w') as f:
        for k in archive.values():
            f.write(str(k.fitness) + ' ')
            write_array(k.centroid, f)
            write_array(k.desc, f)
            write_array(k.x, f)
            f.write("\n")
