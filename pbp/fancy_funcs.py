import numpy as np
BIT_ORDER = 'little'


def bin_encoder(a: np.ndarray, kwargs):
    length = kwargs.get("length", 0)
    byte_size = kwargs.get("byte_size", 8)
    bin_arr_type = kwargs.get("bin_arr_type", np.uint8)
    bitorder = kwargs.get("bitorder", 'little')

    b = np.zeros((length, byte_size), dtype=bin_arr_type)

    for y in range(a.shape[0]):
        b[y, a[y] % byte_size] = 1

    res = np.packbits(b.reshape(-1, byte_size // 8, 8)[:, ::-1], bitorder=bitorder)
    res = res.view(bin_arr_type)

    del b
    return res


def calculate_bin_size(l):
    byte_size = 64
    bin_arr_type = np.uint64

    if l < 4294967295:
        byte_size = 32
        bin_arr_type = np.uint32

    if l < 65535:
        byte_size = 16
        bin_arr_type = np.uint16

    if l < 255:
        byte_size = 8
        bin_arr_type = np.uint8

    return byte_size, bin_arr_type
