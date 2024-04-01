
import numpy as np
import logging
import sys
from bitarray import bitarray, frozenbitarray
from bitarray.util import ba2int, int2ba
import pandas as pd



BIT_ORDER="little"
logging.basicConfig(format='%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG )
logger = logging.getLogger()

np.set_printoptions(threshold=sys.maxsize)
sub_s = "₀₁₂₃₄₅₆₇₈₉"


# Function to encode the binary variables
def bin_encoder(v, l):
    y = bitarray(l, endian=BIT_ORDER)
    y[v] = 1
    return ba2int(y)


# Function to create П matrix
def create_perm(C: np.array):
    perm = C.argsort(kind='quick', axis=0)
    return perm



def calculate_degree(y):
    y_bin = int2ba(y, endian=BIT_ORDER)
    degree = y_bin.count()
    return degree


# Function to create the coefficients using the П matrix
def create_coeffs_matrix(C: np.array, perm: np.array, **kwargs):
    sorted_c = np.take_along_axis(C, perm, axis=0)
    zs = np.zeros((sorted_c.shape[1],), dtype=sorted_c.dtype)

    coeffs_c = np.vstack((sorted_c, zs)) - np.vstack((zs, sorted_c))
    coeffs_c = coeffs_c[:-1]

    return coeffs_c



# Function to create the Boolean variables 
def create_variable_matrix(C: np.array, perm: np.array):
    y = perm[:-1]
    y_maker = np.frompyfunc(bin_encoder, 2, 1)
    
    y = y_maker(y, perm.shape[0])
    y = y.cumsum(axis=0)

    return y



# Function to calculate the term's degree
def calculate_degree(y):
    y_bin = int2ba(y, endian=BIT_ORDER)
    degree = y_bin.count()
    return degree


# Function to reduce the polynomial
def reduce_pbp_pandas(coeffs: np.array, variables: np.array):
    zero_vars = np.zeros((1, variables.shape[1]), dtype=int)
    var_flat = np.vstack([zero_vars, variables]).ravel()

    df = pd.DataFrame()
    
    df["y"] = var_flat
    df["coeffs"] = coeffs.ravel()
        

    df = df.groupby(['y'], as_index=False).agg({'y': 'first', 'coeffs': 'sum' })
    
    zero_coeffs = df["coeffs"] == 0
    df = df.loc[~zero_coeffs]

    df["y_str"] = df["y"].apply(decode_var)

    df["degree"] = df["y"].apply(calculate_degree)
    df.sort_values(by=['degree'], inplace=True)
    blankIndex=[''] * len(df)
    df.index=blankIndex


    return df


# Function to decode Boolean variables
def decode_var(y):
    bin_indices = int2ba(y, endian=BIT_ORDER)
    y_arr  = np.frombuffer(bin_indices.unpack(), dtype=bool)
    indices = np.nonzero(y_arr)[0]

    if indices.size == 0:
        return ""

    return "y" + "y".join([sub_s[i+1] for i in indices])


# Driver function to create a whole pBp
def create_pbp(c: np.array):
    assert len(c.shape) == 2
    perm_c = create_perm(c)
    coeffs_c = create_coeffs_matrix(c, perm_c)
    
    y = create_variable_matrix(c, perm_c)
    pBp = reduce_pbp_pandas(coeffs_c, y)
    
    return pBp



# Function to truncate a pBp by a given p value
def truncate_pBp(pBp, c, p):
    cutoff = c.shape[0] - p + 1
    truncated_pBp = pBp.loc[pBp['degree'] < cutoff]
    return truncated_pBp 

# Function to show terms added together as a polynomial 
def to_string(row):
    return f'{row["coeffs"]}{row["y_str"]}'

def trunc_driver(c, p_list):
    print()
    print()
    pBp = create_pbp(c)
    print("Result pBp")
    polynomial = " + ".join(pBp.apply(to_string, axis=1))
    print(polynomial)
    print("=" * 100)

    for p in p_list:
        truncated_pBp = truncate_pBp(pBp, c, p)
        polynomial = " + ".join(truncated_pBp.apply(to_string, axis=1))
        print(f"p = {p}")
        print(polynomial)
        print("=" * 100)


# Examples 
if __name__ == "__main__":
    c = np.array([
            [7, 8, 2, 10, 3],
            [4, 12, 1, 8, 4],
            [5, 3, 0, 6, 9],
            [9, 6, 7, 1, 5]
        ])

    trunc_driver(c, [2, 3, 4])


    c = np.array([
            [7, 15, 10, 7, 10],
            [10, 17, 4, 11, 22],
            [16, 7, 6, 18, 14],
            [11, 7, 6, 12, 8]
        ])

    trunc_driver(c, [2, 3, 4])


    c = np.array([
            [0, 4, 6, 6, 6, 0, 32, 6],
            [12, 0, 4, 9, 4, 0, 28, 5],
            [9, 2, 0, 3, 3, 0, 20, 3],
            [6, 3, 2, 0, 4, 0, 24, 4],
            [18, 4, 6, 12, 0, 0, 32, 6],
            [15, 4, 4, 9, 5, 0, 12, 1],
            [24, 7, 10, 18, 8, 0, 0, 3],
            [18, 5, 6, 12, 6, 0, 12, 0]
        ])

    trunc_driver(c, [2, 3, 4, 5, 6, 7])


