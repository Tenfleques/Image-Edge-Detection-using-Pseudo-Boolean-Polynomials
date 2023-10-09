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


        