from collections import deque
import timeit
import numpy as np
import tqdm
import pydot
import traceback
import base64
import os
from PIL import Image
from io import BytesIO
import sys
import logging

np.set_printoptions(threshold=sys.maxsize)

logger = logging.getLogger(__file__)


def is_power_of_two(x):
    return (x != 0) and ((x & (x - 1)) == 0)


class PseudoBooleanPolynomials:
    column_counter = 0
    row_counter = 0
    power_lookup = []

    def __init__(self, c: np.ndarray) -> None:
        self.set_c(c)
        self.bit_order = 'little'

    def __reset_column_counter(self):
        self.column_counter = 0

    def __reset_row_counter(self):
        self.row_counter = 0

    def set_c(self, c):
        self.c = c
        self.byte_size, self.bin_arr_type = self.calculate_bin_size(c.shape[0] * c.shape[1])

        self.power_lookup = 2**np.arange(self.byte_size)

    @staticmethod
    def calculate_bin_size(arr_size):
        byte_size = 64
        bin_arr_type = np.uint64

        if arr_size < 255:
            byte_size = 8
            bin_arr_type = np.uint8

        elif arr_size < 65535:
            byte_size = 16
            bin_arr_type = np.uint16

        elif arr_size < 4294967295:
            byte_size = 32
            bin_arr_type = np.uint32

        # print("[INFO] byte size to store decision variables {} {}".format(byte_size, bin_arr_type))

        return byte_size, bin_arr_type

    def set_random_c(self, size=10, start=0, end=10):
        if isinstance(size, int):
            size = (size, size)
        else:
            try:
                r, c = size
            except Exception as err:
                logger.excption(err)
                traceback.print_exc()

        self.set_c(np.random.randint(start, end, size=size))

    def get_permutations(self):
        axis = 0
        return self.c.argsort(kind='quicksort', axis=axis)

    def get_sorted_c(self, permutations: np.ndarray) -> np.ndarray:
        return np.take_along_axis(self.c, permutations, axis=0)

    @staticmethod
    def get_delta_c(sorted_c: np.ndarray) ->np.ndarray:
        zs = np.zeros((sorted_c.shape[1],), dtype=sorted_c.dtype)

        delta_c = np.vstack((sorted_c, zs)) - np.vstack((zs, sorted_c))

        delta_c = delta_c[:-1]

        return delta_c

    @staticmethod
    def get_y_rep(permutations):
        y = permutations + 1
        yy = np.full(y.shape, "",  dtype=str).tolist()

        for i in tqdm.tqdm(range(1, y.shape[0])):
            for j in range(y.shape[1]):
                tmp = yy[i - 1][j] + "y{}".format(y.T[j, i - 1])
                y_indices = sorted(tmp.split('y'))

                yy[i][j] = "y".join(y_indices)

        return np.array(yy)

    def bin_encoder(self, a : np.ndarray, kwargs):
        length = kwargs.get("length", 0)

        b = np.zeros((length, self.byte_size), dtype=self.bin_arr_type)

        for y in range(a.shape[0]):
            b[y, a[y]%self.byte_size] = 1

        res = np.packbits(b.reshape(-1, self.byte_size // 8, 8)[:, ::-1], bitorder=self.bit_order)
        res = res.view(self.bin_arr_type)
        del b
        return res

    def get_y_rep_matrix(self, permutations) -> np.ndarray:
        y = permutations[:-1].copy()

        y = np.apply_along_axis(self.bin_encoder, 0, y, kwargs={"length": y.shape[0]})
        y = y.cumsum(axis=0)

        return y

    @staticmethod
    def shrink_y(a: np.ndarray) -> np.ndarray:
        valid_mask = a != 0
        flipped_mask = valid_mask.sum(1,keepdims=1) > np.arange(a.shape[1]-1,-1,-1)
        flipped_mask = flipped_mask[:,::-1]
        a[flipped_mask] = a[valid_mask]
        a[~flipped_mask] = 0

        a = a[a.any(axis=1)]
        a = a[:, a.any(axis=0)]

        return a

    def __ri_r_minus_1_organiser(self, row):
        rearanged = np.arange(row.shape[0])

        self.column_counter += 1

        if self.column_counter == 1:
            return rearanged

        prev_column = self.column_counter - 2

        linked = row ^ self.reduced_terms[prev_column]
        mask = np.isin(linked, self.power_lookup)

        if mask.all():
            return rearanged

        unlinked = (~mask).nonzero()[0]

        for i in unlinked:
            # test link
            if row[i] ^ self.reduced_terms[prev_column, i - 1]:
                rearanged[i] = i - 1
                rearanged[i - 1] = i

            elif i + 1 < rearanged.shape[0]:
                if row[i] ^ self.reduced_terms[prev_column, i + 1]:
                    rearanged[i] = i + 1
                    rearanged[i + 1] = i

        self.reduced_terms[self.column_counter - 1] = row[rearanged]
        return rearanged

    def __get_best_chains(self):
        return np.apply_along_axis(self.__ri_r_minus_1_organiser, 1, self.reduced_terms)

    def reduce_pbp(self, y: np.array, delta_c: np.array):
        # local aggregation of locations: monomials of zero coeffs are zeroed
        # make first row always out by setting vals = 1
        zeroth_row = delta_c[0].copy()
        # d_c = delta_c.copy()
        delta_c[0] = 1
        empty = np.argwhere(delta_c == 0).T
        delta_c[0] = zeroth_row

        if empty.shape[0]:
            xs, ys = empty
            xs -= 1

            y[xs, ys] = 0

        # group similar monomials
        y_row = y.ravel()
        unq_v, idx_start, count = np.unique(y_row, return_counts=True, return_index=True)

        count -= 1
        non_uniq_counts = count.nonzero()[0]

        unq_v = unq_v[non_uniq_counts]

        flatted_coeffs = delta_c.copy().ravel()
        new_y_row = y_row.copy()

        for dp in unq_v: 
            y_zeroable_cells = np.argwhere(y_row==dp).ravel()
            summable_cells = y_zeroable_cells + y.shape[1]

            if summable_cells.shape[0]:
                flatted_coeffs[summable_cells[-1]] = flatted_coeffs[summable_cells].sum()
                flatted_coeffs[summable_cells[:-1]] = 0
                # zero the monomials summed 
                new_y_row[y_zeroable_cells[:-1]] = 0

        reduced = flatted_coeffs.reshape(delta_c.shape)
        axis = 1
        y_reduced = new_y_row.reshape(y.shape)

        carve = y_reduced.argsort(kind='quicksort', axis=axis)

        y_reduced = np.take_along_axis(y_reduced, carve, axis=axis)
        reduced[1:, :] = np.take_along_axis(reduced[1:, :], carve, axis=axis)

        reduced[0, -1] = reduced[0].sum()
        reduced[0, :-1] = 0

        carve = reduced.any(axis=0)

        self.reduced_coeffs, self.reduced_terms = reduced[:, carve], y_reduced[:, carve]

        # self.__reset_column_counter()
        # self.__reset_row_counter()

        # chained_indices = self.__get_best_chains()

        # self.reduced_coeffs[1:, :] = np.take_along_axis(self.reduced_coeffs[1:, :], chained_indices, axis=1)

        self.reduced_coeffs[0,0] = reduced[0, -1]
        self.reduced_coeffs[0, 1:] = 0

        return self.reduced_coeffs, self.reduced_terms

    def get_chain_cost(self, column):
        return np.sum(column)

    def get_aggregated_chain(self, coeffs, y):
        """
            3. Based on the largest size of an antichain you will design an algorithm covering all terms of pBp by the sub-chains each of which will be represented by the corresponding column in an aggregated matrix.
        """
        print(coeffs)
        print(y)
        # y_indices = self.visualize_monomials(y)

        # print(y_indices)
        aggregated_costs = np.cumsum(coeffs, axis=0)

        return aggregated_costs

    def __diagonal_chain(self):
        """
            finds the chain that goes diagonal apart from the column-wise chains
        """

    def get_chains(self):
        """
            0. You are going to study Dilworth theorem inluding all notions reated to the chain, sub-chain and antichain. For any collection of a partially oredered sets indicated on the Hasse diagram you prepare  a 4-dimensional Boolean at which all mentioned above notions will be explained and illustrated. Finally, we are making a one-to-one mapping between the terms of pBp and collection of sets which should be illustrated by means of a small numerical example.
        """
        # straight_chains = self.reduced_terms.T
        # self.__

        return "", ""

    def get_largest_antichain(self):
        """
            Based on the stored  pBp and Dilworth theorem you either implement or apply your own algorithm for computing the largest size of an antichain supported by a set of non-embedded terms.
        """
        # return chains

    def similar_monomials(self, y: np.ndarray) -> np.ndarray:        
        flatted = y.ravel()
        _, idx_start, count = np.unique(flatted, return_counts=True, return_index=True)
        count -= 1

        dup_vals = idx_start[count.nonzero()[0]]

        return dup_vals

    @staticmethod
    def truncation(coefficients: np.ndarray, y: np.ndarray, p: int) -> np.ndarray:
        return coefficients[:-p+1, :], y[:-p+1, :]

    def get_adjacent_list_wrk(self, yy :np.ndarray) -> np.ndarray:
        print(yy)

        self.anti_chains = self.shrink_y(yy.copy())
        y = self.anti_chains.astype(np.uint64).ravel()
        y = y[y.nonzero()[0]]

        res = np.zeros((y.shape[0], y.shape[0]), np.uint8)

        for i in tqdm.tqdm(range(y.shape[0])):
            for j in range(i):
                r = y[j] ^ y[i]
                r = (r & r - 1) == 0
                if r:
                    res[j, i] = 1
                del r

        xs, ys = res.nonzero()

        adj_list = np.stack([y[xs], y[ys]]).T       

        return adj_list

    def f(self, x, y):
        # create the incidence of entry x, y
        r = self.yy[x] ^ self.yy[y]
        return (r & r - 1) == 0

    @staticmethod
    def onecold(a):
        n = len(a)
        s = a.strides[0]
        strided = np.lib.stride_tricks.as_strided
        b = np.concatenate((a,a[:-1]))

        return strided(b[1:], shape=(n-1,n), strides=(s,s))

    
