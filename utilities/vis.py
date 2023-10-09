import numpy as np
import os

def get_sub(x):
    # create a variable subscript
    x = str(x)
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"

    res = x.maketrans(''.join(normal), ''.join(sub_s))
    return x.translate(res)


def get_monomial_str(bin_row: np.ndarray) -> np.ndarray:

    if bin_row.any():
        jth = np.argwhere(bin_row)
        res = 'y'.join((jth + 1).astype(np.str_).ravel())
    else:
        res = '0'
    return res


def bin_to_str(y_bin_row, latex=False):
    valids = np.nonzero(y_bin_row)[0] + 1
    if len(valids):
        if latex:
            res = "y_" + "y_".join(["{" + str(j) + "}" for j in valids])
        else:
            res = "y" + "y".join([get_sub(j) for j in valids])
    else:
        res = ""

    return res


def bin_to_int(y_bin_row):
    valids = np.nonzero(y_bin_row)[0] + 1
    res = None
    if len(valids):
        res = np.array(valids)

    return res


def get_indices(yy: np.array, byte_size, bit_order="little"):
    if not np.array(yy.shape).all():
        return np.array([])

    stacks = []
    for row in yy:
        r = np.unpackbits(row.view(np.uint8), bitorder=bit_order)
        stacks.append(r)

    stacks = np.hstack(stacks)

    if len(yy.shape) > 1:
        stacks = np.reshape(stacks, (-1, yy.shape[1], byte_size))
    else:
        stacks = np.reshape(stacks, (-1, byte_size))

    # print(stacks)

    res = []
    for i in range(stacks.shape[0]):
        r = []
        for j in range(stacks.shape[1]):
            b_int = bin_to_int(stacks[i, j, :])
            if b_int is not None:
                r.append(b_int)
        if len(r):
            res.append(r)

    print(res)
    # res = np.asarray(res)
    # for i in res:
    #     print(i)

    return res


def visualize_monomials(y: np.ndarray, byte_size, bit_order='little', binary=False, latex=False)-> np.ndarray:
    yy = y.copy()

    if not np.array(yy.shape).all():
        return np.array([])

    stacks = []
    for row in yy:
        r = np.unpackbits(row.view(np.uint8), bitorder=bit_order)
        stacks.append(r)

    stacks = np.hstack(stacks)

    if len(yy.shape) > 1:
        stacks = np.reshape(stacks, (-1, yy.shape[1], byte_size))
    else:
        stacks = np.reshape(stacks, (-1, byte_size))

    # print(stacks)

    if binary:
        return stacks

    res = []
    for i in range(stacks.shape[0]):
        r = []
        for j in range(stacks.shape[1]):
            r.append(bin_to_str(stacks[i, j, :], latex=latex))
        res.append(r)

    # print(res)
    res = np.asarray(res)
    first_row = np.full((1, res.shape[1]), '')

    return np.vstack([first_row, res])


def visualize_polynomial(k: np.ndarray, y: np.ndarray, byte_size: int, bit_order: str, latex=False):
    if k.shape[0] * k.shape[-1] < 400:
        y_human = visualize_monomials(y, byte_size=byte_size, bit_order=bit_order, latex=latex)
        zx, zy = np.argwhere(k == 0).T
        monoms_human = np.array([])

        try:
            if len(y_human):
                monoms_human = np.core.defchararray.add(k.astype(np.str_), y_human)
            else:
                monoms_human = k.astype(np.str_)
        except Exception as err:
            print(err)

        return monoms_human

    # too large to use latex 
    return "Data too large"


if __name__ == "__main__":
    y = np.array([
        [256, 257],
        [3840, 19968]
    ], dtype=np.uint16)

    byte_size = 16
    bin_arr_type = np.uint16

    # res = []
    # for row in y:
    #     res.append(np.unpackbits(row.view(np.uint8)))

    # res = np.array(res)
    # print(res)

    # print("\n\n")
    # res = res.reshape(-1, y.shape[1], byte_size)
    # print(res)
    y_vis = visualize_monomials(y, byte_size, latex=True)

    print(y_vis)
