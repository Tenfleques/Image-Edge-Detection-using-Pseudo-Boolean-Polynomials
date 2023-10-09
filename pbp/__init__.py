# import os 
# import sys
# import numpy as np
# import argparse
# import logging 

# from gen_tools.graph_utils import read_graph
# from pbp import PseudoBooleanPolynomials
# from chains_ctrl import ChainsCtrl
# from reducer import PBPReducer, calculate_bin_size, bin_encoder, p_truncation, get_perm_diff_matrices
# from gen_tools.presentation import step_cb, visualize_monomials, visualize_polynomial, hasse_digram


# logger = logging.getLogger(__name__)
# BIT_ORDER = 'little'

# class PBPCtrl:
#     def __init__(rows=0, cols=0, data=None, source=""):
#         self.data = None 

#         c = np.array([
#             [7, 15, 10, 7, 10],
#             [10, 17, 4, 11, 22],
#             [16, 7, 6, 18, 14],
#             [11, 7, 6, 12, 8]
#         ])

#         self.pseudo_boolean_polynomials = PseudoBooleanPolynomials(c)

#         if data is not None:
#             self.data = np.array(data)
#             self.pseudo_boolean_polynomials.set_c(self.data.copy())
#         elif source:
#             try:
#                 self.data = read_graph(source)
#                 self.pseudo_boolean_polynomials.set_c(self.data.copy())

#             except Exception as err:
#                 logger.exception("failed to read graph {}".format(err))
#         if rows is not None and cols is not None:
#             try:
#                 rows = int(rows)
#                 cols = int(cols)
#                 max_val = int(max_val)
#             except Exception as exc:
#                 step_cb("[ERROR] {} \n using default matrix \n".format(exc))
#                 rows = 6
#                 cols = 10
#                 max_val = 255
#             else:
#                 self.pseudo_boolean_polynomials.set_random_c((rows, cols), 0, max_val) 





#     def get_perm_matrix(self, sort_algo="quicksort"):
#         perm = data.argsort(kind='quicksort', axis=0)
#         return perm


#     def get_sorted_matrix(self):
#         pass


# def test_main():
#     data = np.array([
#         [ 8, 8, 8, 5],
#         [ 12, 7, 5, 7],
#         [ 18, 2, 3, 1 ],
#         [ 5, 18, 9, 8 ]
#     ])
#     p = 0


# if __name__ == "__main__":
#     # if not args 
#     test_main()