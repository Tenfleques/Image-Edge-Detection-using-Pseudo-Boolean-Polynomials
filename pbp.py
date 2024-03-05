import numpy as np
import sys
import os
import json
import logging
import itertools
from bitarray import bitarray, frozenbitarray
from bitarray.util import ba2int, int2ba
import pandas as pd
import glob
import time
from datetime import datetime

root_dir = os.path.dirname(__file__)


BIT_ORDER="little"
logging.basicConfig(format='%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG if os.getenv('PBP_DEBUG', None) else logging.INFO)
logger = logging.getLogger(__file__)

np.set_printoptions(threshold=sys.maxsize)

def bin_encoder(v, l):
    x = bitarray(l, endian=BIT_ORDER)
    x[v] = 1
    return ba2int(x)

def calculate_degree(y):
    y_bin = int2ba(y, endian=BIT_ORDER)
    degree = y_bin.count()
    return degree

def create_perm(C: np.array):
    perm = C.argsort(kind='quick', axis=0)
    return perm

def create_coeffs_matrix(C: np.array, perm: np.array, **kwargs):
    sorted_c = np.take_along_axis(C, perm, axis=0)
    zs = np.zeros((sorted_c.shape[1],), dtype=sorted_c.dtype)

    coeffs_c = np.vstack((sorted_c, zs)) - np.vstack((zs, sorted_c))
    coeffs_c = coeffs_c[:-1]

    return coeffs_c


def reduce_pbp_pandas(coeffs: np.array, variables: np.array):
    zero_vars = np.zeros((1, variables.shape[1]), dtype=int)
    var_flat = np.vstack([zero_vars, variables]).ravel()

    df = pd.DataFrame()
    
    df["coeffs"] = coeffs.ravel()
    df["y"] = var_flat    

    df = df.groupby(['y']).agg({'coeffs': 'sum', 'y': 'first', })
    
    df["degree"] = df["y"].apply(calculate_degree)
    df["id"] = np.arange(df.shape[0])
    
    zero_coeffs = df["coeffs"] == 0
    df = df.loc[~zero_coeffs]
    
    # print(df)
    
    return df


def create_variable_matrix(C: np.array, perm: np.array):
    y = perm[:-1]
    logger.debug(f"Y from perm matrix {y.dtype}")
    
    y_maker = np.frompyfunc(bin_encoder, 2, 1)
    y = y_maker(y, perm.shape[0])
    
    logger.debug(f"Y bin encoded type {y.dtype}")
    y = y.cumsum(axis=0)

    return y


def create_pbp(c: np.array):
    assert len(c.shape) == 2
    perm_c = create_perm(c)
    coeffs_c = create_coeffs_matrix(c, perm_c)
    
    y = create_variable_matrix(c, perm_c)
    pBp = reduce_pbp_pandas(coeffs_c, y)
    
    return pBp


def truncate_pBp(pBp, p):
    truncated_pBp = pBp.loc[pBp['degree'] < p]
    return truncated_pBp 
    

if __name__ == "__main__":
    c = np.array([
        [7, 8, 2, 10, 3],
        [4, 12, 1, 8, 4],
        [5, 3, 0, 6, 9],
        [9, 6, 7, 1, 5]
    ])

    pBp = create_pbp(c)
    print(pBp)
