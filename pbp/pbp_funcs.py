import numpy as np 
import pandas as pd
import timeit
import os
import subprocess
import pydot
from IPython.display import Latex, SVG, Math

# function to convert to subscript
def get_sub(x):
    x = str(x)
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    res = x.maketrans(''.join(normal), ''.join(sub_s))
    return x.translate(res)
    

def bin_to_str(y_bin_row):
    valids = np.nonzero(y_bin_row)[0] + 1
    res = "y" + "y".join([get_sub(j) for j in  valids])
    # print(res)
    return res

def visualize_monomials(y: np.ndarray, byte_size, bit_order='little', binary=False)-> np.ndarray:
    yy = y.copy()

    if not np.array(yy.shape).all():
        return np.array([])

    stacks = []
    for i in range(byte_size//8):
        yy = yy + (1 << 8)

        un_packed = np.unpackbits(yy.astype(np.uint8), axis=1, bitorder=bit_order)

        un_packed = np.reshape(un_packed, (-1, 8))
        
        stacks.append(un_packed)

    
    stacks = np.hstack(stacks)
    if len(yy.shape) > 1:
        stacks = np.reshape(stacks, (-1, yy.shape[1], byte_size))
    else:
        stacks = np.reshape(stacks, (-1, byte_size))
    
    if binary:
        return stacks

    res = []
    for i in range(stacks.shape[0]):
        r = []
        for j in range(stacks.shape[1]):
            r.append(bin_to_str(stacks[i, j, :]))
            
        res.append(r)
    
    res = np.asarray(res)
    first_row = np.full((1, res.shape[1]), '')
    
    return np.vstack([first_row, res])

def visualize_polynomial(k: np.ndarray, y: np.ndarray, byte_size: int, bit_order: str):
    if k.shape[0] * k.shape[-1] < 100:
        y_human = visualize_monomials(y, byte_size=byte_size, bit_order=bit_order)
        zx, zy = np.argwhere(k==0).T
        y_human[zx, zy] = ""
        monoms_human = np.core.defchararray.add(k.astype(np.str_), y_human)
        return monoms_human
    
    # too large to use latex 
    return "Data too large"


def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').replace("'", "").splitlines()
    rv = [r'\begin{bmatrix}']
    rv += [' ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    
    return Math('\n'.join(rv))

def step_cb(d):
    if isinstance(d, np.ndarray):
        if len(d.shape) <= 2:
            if d.shape[0] * d.shape[-1] <= 200:
                return bmatrix(d)
    print(d)

def bin_encoder(a : np.ndarray, kwargs):
    length = kwargs.get("length", 0)
    byte_size = kwargs.get("byte_size", 8)
    bin_arr_type = kwargs.get("bin_arr_type", np.uint8)
    bitorder = kwargs.get("bitorder", 'little')
    
    b = np.zeros((length, byte_size), dtype=bin_arr_type)

    for y in range(a.shape[0]):
        b[y, a[y]%byte_size] = 1

    res = np.packbits(b.reshape(-1, byte_size//8, 8)[:, ::-1], bitorder=bitorder).view(bin_arr_type)

    del b
    return res


def calculate_bin_size(l):
    byte_size = 64
    bin_arr_type = np.uint64
    # print(l)

    if l < 255:
        byte_size = 8
        bin_arr_type = np.uint8

    elif l < 65535:
        byte_size = 16
        bin_arr_type = np.uint16

    elif l < 4294967295:
        byte_size = 32
        bin_arr_type = np.uint32

    return byte_size, bin_arr_type

class PBPReducer:    
    def __init__(self, byte_size):
        self.column_counter = 0
        self.row_counter = 0
        self.power_lookup = 2**np.arange(byte_size)

    def __reset_column_counter(self):
        self.column_counter = 0
    
    def __reset_row_counter(self):
        self.row_counter = 0

    @staticmethod
    def reduce_column(column: np.ndarray, y_col: np.ndarray):
        _, idx_start, count = np.unique(y_col, return_counts=True, return_index=True)
        count -= 1

        dup_vals = y_col[idx_start[count.nonzero()[0]]]
        empty_coeffs = np.argwhere(column==0)

        if empty_coeffs.shape[0]:
            y_col[empty_coeffs[0]] = 0
        
        for dp in dup_vals:
            reduce_cols = np.argwhere(y_col==dp)
            column[reduce_cols[-1]] = column[reduce_cols].sum()
            column[reduce_cols[:-1]] = 0

        y_col[idx_start[count.nonzero()[0]]] = 0

        return column, y_col


    def __shrink_monomials(self):
        # nullify zero counterparts 
        # zeros = np.argwhere(self.reduced_coeffs == 0)
        # if np.array(zeros.shape).all():
        #     zs_x, zs_y = zeros.T
        #     self.reduced_terms[zs_x, zs_y] = 0

        # zeros = np.argwhere(self.reduced_terms == 0)
        # if np.array(zeros.shape).all():
        #     zs_x, zs_y = zeros.T
        #     self.reduced_coeffs[zs_x, zs_y] = 0

        # delete empty columns 
        carve = self.reduced_coeffs.any(axis=0)
        # print("old shape before truncating zero rows and zero columns ", self.reduced_coeffs.shape)

        self.reduced_coeffs, self.reduced_terms = self.reduced_coeffs[:, carve], self.reduced_terms[:, carve]

        # delete empty rows
        carve_rows = self.reduced_coeffs.any(axis=1)

        self.reduced_terms = self.reduced_terms[carve_rows, :]
        self.reduced_coeffs = self.reduced_coeffs[carve_rows, :]

        # print("new shape after truncating zero rows and zero columns ", self.reduced_coeffs.shape)


    def reduce_pbp(self, y: np.ndarray, delta_c: np.ndarray) -> np.ndarray:
        delta_c = delta_c.copy()
        y = y.copy()

        constant_term = delta_c[0].sum()
        delta_c[0, 1:] = 0        

        for i, col in enumerate(delta_c[1:]):
            delta_c[1+i], y[i] = self.reduce_column(col, y[i])

        yy = y.argsort(kind='quicksort', axis=1)

        self.reduced_terms = np.take_along_axis(y, yy, axis=1)
        self.reduced_coeffs = np.take_along_axis(delta_c[1:, :], yy, axis=1)
        

        self.__shrink_monomials()

        self.__reset_column_counter()
        self.__reset_row_counter()


        # restore first row
        first_row = np.zeros((self.reduced_coeffs.shape[1]), dtype=self.reduced_coeffs.dtype)
        first_row[0] = constant_term
        # print(first_row[0])

        self.reduced_coeffs = np.vstack([first_row, self.reduced_coeffs])

        return self.reduced_coeffs, self.reduced_terms


class ChainsCtrl:
    def __init__(self, y, byte_size, bit_order='little'):
        self.power_lookup = 2**np.arange(byte_size)
        self.y = y.copy()
        self.row_counter = 0
        self.col_counter = 0
        y1 = self.y[:-1, :]
        y2 = self.y[1:, :]

        self.chains = None
        self.anti_chains = None
    
    def __reset_counters(self):
        self.row_counter = 0
        self.col_counter = 0

    def __get_chains(self, y1, y2):
        column_chains = np.isin(y1 ^ y2, self.power_lookup).all(axis=0)
        return np.vstack([y1[:-1,column_chains], y2[:, column_chains]])

    def __get_anti_chains(self, y1, y2):
        # print(y1 ^ y2)
        column_chains = np.isin(y1 ^ y2, self.power_lookup, invert=True).all(axis=0)
        return np.vstack([y1[:-1,column_chains], y2[:, column_chains]])
    
    def __acc_chains(self, r, y1):
        for j in range(0, self.y.shape[1]):
            y1[self.row_counter] = np.roll(self.y[self.row_counter], 1)
            ch = self.__get_chains(y1[:-1, :], y1[1:, :])

            if self.chains is None:
                self.chains = ch
            else:
                if np.array(ch.shape).all():
                    self.chains = np.hstack([self.chains, ch])

        self.row_counter += 1

    def __acc_anti_chains(self, r):
        is_anti_chain = np.isin(self.chains.T[self.row_counter, 1:] ^ self.chains.T[:, :-1], self.power_lookup, invert=True)
        
        print(self.chains.T[self.row_counter])
        print(is_anti_chain)

        print()

        if is_anti_chain.all():
            print(self.chains.T[self.row_counter])
            
            # bin_decider.append(is_anti_chain)

        # bin_decider = np.array(bin_decider)
        # print(np.where(bin_decider))

        self.row_counter += 1

    def fast_anti_chains(self):
        self.__reset_counters()
        if self.chains is None:
            self.fast_chains()

        np.apply_along_axis(self.__acc_anti_chains, 0, self.chains)

        # print(self.anti_chains)

        return self.anti_chains

        anti_chains = np.unique(self.anti_chains, axis=0)

        zeros = np.zeros((2, anti_chains.shape[1]), dtype=anti_chains.dtype)
        last_row = (2**np.arange(anti_chains.shape[0] + 1)).sum()

        zeros[1] = last_row

        return np.vstack([zeros[0], anti_chains, zeros[1]])
    
    def fast_chains(self):
        self.__reset_counters()
        y1 = self.y      
        np.apply_along_axis(self.__acc_chains, 1, self.y, y1)
    
        self.chains = np.unique(self.chains, axis=0)
        self.chains = np.unique(self.chains, axis=1)

        zeros = np.zeros((2, self.chains.shape[1]), dtype=self.chains.dtype)
        last_row = (2**np.arange(self.chains.shape[0] + 1)).sum()

        zeros[1] = last_row

        return np.vstack([zeros[0], self.chains, zeros[1]])

    def __collect_cover_chain(self, row):
        if self.cover_chain is None:
            self.cover_chain = row
        else:
            x = np.any(self.cover_chain == row)
            if not x:
                self.cover_chain = np.vstack([self.cover_chain, row])
        
        return row

    def get_cover_chain(self, chains):
        self.cover_chain = None
        np.apply_along_axis(self.__collect_cover_chain, 0, chains)

        return self.cover_chain


    @staticmethod
    def __column_to_bin_nodes(c):
        l = c.shape[0]
        # print(c)
        bin_list = np.zeros((l-1, 2), dtype=c.dtype)
        for i in range(l-1):
            bin_list[i] = c[i], c[i+1]

        # print(bin_list)

        return bin_list

    def get_adj_list(self, chains):
        if np.array(chains.shape).all():
            return np.apply_along_axis(self.__column_to_bin_nodes, 0, chains).transpose(0, 2, 1).reshape(-1, 2)

        return np.array([])


def get_monomial_str(bin_row: np.ndarray) -> np.ndarray:
    if bin_row.any():
        jth = np.argwhere(bin_row)
        res = ''.join((jth+1).astype(np.str_).ravel()) 
    else:
        res = '0'
    return res


def hasse_digram(bin_adj_list, int_adj_list=None, out_path=""):
    res = [ 'digraph G { rankdir = LR; ' ]
    x = set()

    k = 0
    for bin_rows in bin_adj_list:
        nodes = []
        j = 0
        for bin_str in bin_rows:
            mon_str = get_monomial_str(bin_str)
            
            if int_adj_list is not None:
                # print(int_adj_list[k])
                mon_str = "{}..{}".format(mon_str, int_adj_list[k, j])

            nodes.append(mon_str)
            j += 1

        k += 1
        if len(nodes) == 2:
            if all(nodes) and not (nodes[0] == nodes[1]):
                x.add('y{}->y{};'.format(*nodes))
    
    res += list(x)
    res.append('}')
    
    graphs = pydot.graph_from_dot_data('\n'.join( res ))
    if out_path:
        graphs[0].write_svg(out_path)
    
    return SVG(graphs[0].create_svg())

def xy_incidence(yy, x, y):
    # create the incidence of entry x, y
    r = yy[x] ^ yy[y]
    return (r & r - 1) == 0

def get_adjacent_list(yy :np.ndarray) -> np.ndarray:
    yy = yy.astype(np.uint64).ravel()
    yy = yy[yy.nonzero()[0]]
    
    yy.sort()
    res = np.zeros((yy.shape[0], yy.shape[0]), np.uint8)

    adj_list = []

    indices = np.arange(yy.shape[0])

    start_time = timeit.default_timer()
    res_x = xy_incidence(yy, *np.meshgrid(indices, indices, sparse=True, copy=False))
    res_x = np.tril(res_x, k=-1)
    chains = np.argwhere(res_x == True)

    ys, xs = chains.T

    print("xs, ys", np.stack([xs, ys]))

    adj_list = np.stack([yy[xs], yy[ys]]).T
    adj_list = np.unique(adj_list,  axis=0)
    
    adj_list = adj_list[adj_list.all(axis=1)]
    adj_list = adj_list[(adj_list[:,0] - adj_list[:, 1]).nonzero()[0]]

    return adj_list

def p_truncation(coefficients: np.ndarray, y: np.ndarray, p: int) -> np.ndarray:
    if p > 0 and p < coefficients.shape[0] + 1:
        return coefficients[:p+1, :], y[:p, :]

    return coefficients, y

def bnb_calc(aggregated_costs, verbose=False):
    # pass aggregated costs to bnb 
    os.makedirs("./tmp", exist_ok=True)

    in_bin = "./tmp/reduced-{}.npy".format(timeit.default_timer())
    out_bin = "./tmp/reduced-bnb-{}".format(timeit.default_timer())

    np.save(in_bin, aggregated_costs)
    #%%
    # call bnb 
    p = subprocess.Popen("./bnb/bnb.exe {} -1 {} {}".format(in_bin, out_bin, "-v" if verbose else ""), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    (output, err) = p.communicate()  

    if verbose:
        step_cb("STDOUT")
        step_cb(output.decode("utf-8"))

    retval = p.wait()        
    if retval == 0:
        res = np.fromfile(out_bin + ".bin", dtype=np.int32)
        res = res.reshape(res.shape[0]//3, 3)
        return res
    else:
        step_cb("STDERR")
        step_cb(err.decode("utf-8"))
        return []
        