import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from fancy_funcs import bin_encoder, calculate_bin_size, BIT_ORDER

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class PBPReducer:
    def __init__(self, byte_size):
        self.column_counter = 0
        self.row_counter = 0
        self.power_lookup = 2**np.arange(byte_size)
        self.byte_size = byte_size

    def __reset_column_counter(self):
        self.column_counter = 0

    def __reset_row_counter(self):
        self.row_counter = 0

    def __shrink_monomials(self):
        # delete zero coeffs
        zsc = np.where(self.reduced_coeffs == 0)
        zst = np.where(self.reduced_terms == 0)

        self.reduced_terms[zsc] = 0
        self.reduced_coeffs[zst] = 0

        # print("[INFO] original instances")
        # print(self.reduced_coeffs)
        # print(self.reduced_terms)
        # print("*" * 40)

        # delete empty columns based on coefficients
        carve = self.reduced_coeffs.any(axis=0)
        self.reduced_coeffs = self.reduced_coeffs[:, carve]
        self.reduced_terms = self.reduced_terms[:, carve]

        # print("[INFO] columns reduced based on coeffs")
        # print(self.reduced_coeffs.shape)
        # print(self.reduced_terms.shape)
        # print("*" * 40)

        # delete empty columns based on terms
        carve = self.reduced_terms.any(axis=0)
        self.reduced_coeffs = self.reduced_coeffs[:, carve]
        self.reduced_terms = self.reduced_terms[:, carve]

        # print("[INFO] columns reduced based on terms")
        # print(self.reduced_coeffs)
        # print(self.reduced_terms)
        # print("*" * 40)

        # delete empty rows based on coefficients
        # carve_rows = self.reduced_coeffs.any(axis=1)
        # self.reduced_terms = self.reduced_terms[carve_rows, :]
        # self.reduced_coeffs = self.reduced_coeffs[carve_rows, :]

        # delete empty rows based on terms
        # carve_rows = self.reduced_terms.any(axis=1)
        # print("[INFO] rows reduced based on coeffs", carve_rows)
        # print(self.reduced_coeffs)
        # print(self.reduced_terms)
        # print("*" * 40)

        # self.reduced_terms = self.reduced_terms[:, carve_rows]
        # self.reduced_coeffs = self.reduced_coeffs[:, carve_rows]

    @staticmethod
    def reduce_column(column: np.ndarray, y_col: np.ndarray):
        uniqs, idx_start, count = np.unique(y_col, return_counts=True, return_index=True)
        count -= 1

        dup_vals = y_col[idx_start[count.nonzero()[0]]]
        # print(column, y_col, dup_vals)

        empty_coeffs = np.argwhere(column == 0)

        if empty_coeffs.shape[0]:
            y_col[empty_coeffs[0]] = 0

        for dp in dup_vals:
            reduce_cols = np.argwhere(y_col==dp)

            column[reduce_cols[0]] = column[reduce_cols].sum()
            column[reduce_cols[1:]] = 0

            y_col[reduce_cols[1:]] = 0

        return column, y_col

    def reduce_pbp(self, y: np.ndarray, delta_c: np.ndarray, shrink=True) -> np.ndarray:
        delta_c = delta_c.copy()
        y = y.copy()

        constant_term = delta_c[0].sum()
        delta_c[0, 1:] = 0

        for i, col in enumerate(delta_c[1:]):
            delta_c[1+i], y[i] = self.reduce_column(col, y[i])

        yy = y.argsort(kind='quicksort', axis=1)

        self.reduced_terms = np.take_along_axis(y, yy, axis=1)
        self.reduced_coeffs = np.take_along_axis(delta_c[1:, :], yy, axis=1)

        if shrink:
            self.__shrink_monomials()

        self.__reset_column_counter()
        self.__reset_row_counter()

        # print("first row is ", first_row, constant_term)
        if self.reduced_coeffs.shape[1]:
            # restore first row
            first_row = np.zeros((self.reduced_coeffs.shape[1]), dtype=self.reduced_coeffs.dtype)
            first_row[-1] = constant_term
            self.reduced_coeffs = np.vstack([first_row, self.reduced_coeffs])
        else:
            self.reduced_coeffs = np.array([[constant_term]])

        return self.reduced_coeffs, self.reduced_terms


def get_perm_diff_matrices(data):
    perm = data.argsort(kind='quicksort', axis=0)
    sorted_c = np.take_along_axis(data, perm, axis=0)
    zs = np.zeros((sorted_c.shape[1],), dtype=sorted_c.dtype)
    delta_c = np.vstack((sorted_c, zs)) - np.vstack((zs, sorted_c))
    delta_c = delta_c[:-1]

    return perm, delta_c


def get_monomial_str(bin_row: np.ndarray) -> np.ndarray:
    if bin_row.any():
        jth = np.argwhere(bin_row)
        res = ''.join((jth + 1).astype(np.str_).ravel())
    else:
        res = '0'
    return res


def p_truncation(coefficients: np.ndarray, y: np.ndarray, p: int) -> np.ndarray:
    if p > -1 and p < coefficients.shape[0] + 1:
        return coefficients[:p + 1, :], y[:p, :]

    return coefficients, y


def pbp_instance(data, BIT_ORDER=BIT_ORDER, shrink=False, p=-1, return_byte_size=False):
    global index_counter
    perm = data.argsort(kind='quicksort', axis=0)
    sorted_c = np.take_along_axis(data, perm, axis=0)

    zs = np.zeros((sorted_c.shape[1],), dtype=sorted_c.dtype)
    delta_c = np.vstack((sorted_c, zs)) - np.vstack((zs, sorted_c))
    delta_c = delta_c[:-1]

    byte_size, bin_arr_type = calculate_bin_size(data.shape[0] * data.shape[1])

    y = perm[:-1].astype(bin_arr_type)

    y = np.apply_along_axis(bin_encoder, 0, y, kwargs={"length": y.shape[0], "byte_size": byte_size, "bin_arr_type": bin_arr_type, "bitorder": BIT_ORDER})

    y = y.cumsum(axis=0).astype(bin_arr_type)

    pbp_red = PBPReducer(byte_size=byte_size)
    reduced, reduced_y = pbp_red.reduce_pbp(y, delta_c, shrink=shrink)

    if p > 1:
        return p_truncation(reduced, reduced_y, p, return_byte_size=return_byte_size)

    if return_byte_size:
        return reduced, reduced_y, bin_arr_type

    return reduced, reduced_y


def getPBPRank(y, b_size):
    rank = 0
    if np.array(y.shape).any():
        rank_view = np.array([np.max(y)], dtype=b_size)
        rank_view = np.unpackbits(rank_view.view(np.uint8), bitorder=BIT_ORDER)
        try:
            rank_view = np.argwhere(rank_view)
            if rank_view.shape[0]:
                rank = np.max(rank_view)
        except Exception as err:
            print(err)

    return rank


if __name__ == "__main__":
    c = np.array([
        [127, 127, 127, 127, 127, 127, 127, 127, 127, 127,],
        [127, 127, 127, 127, 127, 127, 127, 127, 127, 127,],
        [127, 127, 127, 127, 127, 127, 127, 127, 127, 127,],
        [127, 127, 127, 127, 127, 127, 127, 127, 127, 127,],
        [127, 127, 127, 127, 127, 127, 127, 127, 127, 127,],
        [127, 127, 127, 127, 127, 127, 127, 127, 127, 127,],
        [127, 127, 127, 127, 127, 127, 127, 127, 127, 127,],
        [127, 127, 127, 127, 127, 127, 127, 127, 127, 127,],
        [127, 127, 127, 127, 127, 127, 127, 127, 127, 127,],
        [127, 127, 127, 127, 127, 127, 127, 127, 127, 127,]
    ])

    coeff, y, bin_size = pbp_instance(c, return_byte_size=True, shrink=True)
    print(coeff, y)
    # r = getPBPRank(y, bin_size)

    # print(r)
