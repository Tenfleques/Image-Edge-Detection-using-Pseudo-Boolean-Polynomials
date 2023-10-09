import cv2
import sys
from pbp import PseudoBooleanPolynomials
import numpy as np
import os
import argparse
import glob 


def main(images="./images"):
    png_images = glob.glob(os.path.join(images, "/*.png"))
    jpg_images = glob.glob(os.path.join(images, "/*.jpg"))

    images = png_images.extend(jpg_images)

    print(images)

def polynomial_sorter(mon_a, var_symbol = "y") -> np.array: 
    # sort by variables 
    split_mon_a = mon_a.split(var_symbol)
    var_part_a = var_symbol.join(split_mon_a[1:])

    return var_part_a



def get_pbp_str(c, p, step_cb=lambda x: None, reorder={}):
    pseudo_boolean_polynomials = PseudoBooleanPolynomials(c)
    permutations = pseudo_boolean_polynomials.get_permutations()

    if len(reorder.keys()):
        for k, v in reorder.items():
            permutations[:, k] = np.array(v)

        print(permutations + 1)
        print()

    sorted_c = pseudo_boolean_polynomials.get_sorted_c(permutations)

    delta_c = pseudo_boolean_polynomials.get_delta_c(sorted_c)


    del sorted_c

    y_str = pseudo_boolean_polynomials.get_y_rep(permutations)
    bc_expression = np.char.add(delta_c.T.astype(str), y_str.T)
    m_c = bc_expression.flatten()
    reduced = pseudo_boolean_polynomials.reduce_similar_monomials(y_str, delta_c)

    # step_cb(" + ".join(reduced))

    # check reduction factor after combinining similar items
    reduction_pecentage = 100 * (len(m_c) - len(reduced))/len(m_c)
    step_cb("[INFO] percentage reduction by adding similar monomials {}%".format(reduction_pecentage))

    print(len(reduced))

    pbp_trunc_2 = pseudo_boolean_polynomials.truncation_str(reduced, p)
    
    print(len(m_c))
    
    str_reduced = sorted(pbp_trunc_2, key=polynomial_sorter)

    # print(len(str_reduced))

    return " + ".join(str_reduced)


def test():
    c1 = np.array([
        [7, 15, 10, 7, 10],
        [10, 17, 4, 11, 22],
        [16, 7, 6, 18, 14],
        [11, 7, 6, 12, 8]
    ])

    c2 = np.array([
        [5, 18, 9, 9, 3],
        [10, 20, 7, 11, 12],
        [15, 10, 9, 15, 3],
        [10, 8, 9, 13, 7]
    ])


    pbp_c1 = get_pbp_str(c1, p=2, step_cb=print)
    pbp_c2 = get_pbp_str(c2, p=2, step_cb=print, reorder={2: [1, 2, 3, 0], 4: [2, 0, 3, 1]})

    print(pbp_c1)
    print()
    print(pbp_c2)
    print()
    print(pbp_c1 == pbp_c2)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--images",
    #     help="image directory", type=str, default="./images")

    ap.add_argument("-t", "--test",
        help="test equivalence instances", dest='test', action='store_true')

    kwargs = vars(ap.parse_args())

    if kwargs.get("test"):
        test()
    