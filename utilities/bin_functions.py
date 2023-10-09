import numpy as np
import logging
BIT_ORDER = 'little'

logging.basicConfig(format='%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)


def is_power_two(x):
    return (x and (not (x & (x - 1))))


# def is_power_of_two(x):
#     return (x != 0) and ((x & (x - 1)) == 0)

vector_is_power_two = np.vectorize(is_power_two)
    

def bin_encoder(a: np.ndarray, **kwargs):
    length = kwargs.get("length", 0)
    byte_size = kwargs.get("byte_size", 8)
    bin_arr_type = kwargs.get("bin_arr_type", np.uint8)
    bitorder = kwargs.get("bitorder", BIT_ORDER)

    b = np.zeros((length, byte_size), dtype=bin_arr_type)

    for y in range(a.shape[0]):
        b[y, a[y] % byte_size] = 1

    res = np.packbits(b.reshape(-1, byte_size // 8, 8)[:, ::-1], bitorder=bitorder)
    res = res.view(bin_arr_type)

    del b
    return res


def calculate_bin_size(arr_size):
    byte_size = 64
    bin_arr_type = np.uint64

    if arr_size < 2**8:
        byte_size = 8
        bin_arr_type = np.uint8

    elif arr_size < 2**16:
        byte_size = 16
        bin_arr_type = np.uint16

    elif arr_size < 2**32:
        byte_size = 32
        bin_arr_type = np.uint32

    logger.debug("byte size to store decision variables {} {}".format(byte_size, bin_arr_type))

    return byte_size, bin_arr_type


def getPBPDegree(y, b_size):
    rank = 0

    if y.size:
        rank_view = np.array([np.max(y)], dtype=b_size)
        rank_view = np.unpackbits(rank_view.view(np.uint8), bitorder=BIT_ORDER)
        try:
            rank_view = np.argwhere(rank_view)
            if rank_view.shape[0]:
                rank = np.max(rank_view)
        except Exception as err:
            print(err)

    return rank