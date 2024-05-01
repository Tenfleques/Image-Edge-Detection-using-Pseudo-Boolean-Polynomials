
# # Example of usage of pseudo-Boolean polynomials applied on image patches to detect edges
# Import libraries


import sys
import os
import cv2
import numpy as np
import json
import pandas as pd
import time
from matplotlib import pyplot as plt

# Import pBp creation functions
from pbp_driver import create_perm, create_coeffs_matrix, create_variable_matrix, BIT_ORDER
# Import image utilities
from image_utils.im_filters import group_ranges, pixelate_bin, kmeans_color_quantization, image_gaussian, resize_img
# Support function for showing an image
def show_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')  # Turn off axis
    plt.show()

# Function to calculate the degrees of terms in a pBp


def deg_greater_than(numbers, p):
    binary_numbers = np.array([bin(num).count('1') for num in numbers])
    deg_greater = np.any(binary_numbers >= p)

    return deg_greater

# 1. Create a variable and coeffs matrix.
# 2. Nullify variables corresponding to the coefficients==0 of the polynomial.
# 3. Calculate the degrees of the variables left

def process_degs(c, p):
    perm_c = create_perm(c)
    coeffs_c = create_coeffs_matrix(c, perm_c)

    y = create_variable_matrix(c, perm_c)
    zs = np.where(coeffs_c[1:] == 0)
    y[zs] = 0
    # no need to calculate whole pBp since we are only interested in the degree
    y = y.ravel()
    deg = deg_greater_than(y[y>0], p)
    return deg

# Function to annotate a high degree patch
def annotate_mask(mask, i, step_y, j, step_x, value=255):
    # cv2.rectangle(mask, (i, j), (i + step_y, j + step_x), 250, -1)
    mask[i, j] = value

# Iterate over image patches and calculate degree of polynomial to compare with Threshold p
def get_patch_degs(img_grey, p, n, m, step_x=2, step_y=2, rotation=False):
    windows = []
    mask = np.full(img_grey.shape, 0, np.uint8)
    h, w = img_grey.shape[:2]

    max_i = h - n + 1
    for i in range(0, max_i, step_y):
        for j in range(0, w - m + 1, step_x):
            i_n = i + n
            j_m = j + m

            window = img_grey[i:i_n, j:j_m]
            if not window.any():
                continue

            deg = process_degs(window, p)
            if deg:
                annotate_mask(mask, i, step_y, j, step_x)
                j += step_x
            elif rotation:
                deg = process_degs(window.T, p)
                if deg:
                    annotate_mask(mask, i, step_y, j, step_x)
                    j += step_x
    return mask



# Driver function
def process_file(im_p, **kwargs):
    if not os.path.exists(im_p):
        return

    print("[INFO] processing file {}".format(im_p))
    target_mask = ".".join(im_p.replace("images", "masks").split(".")[:-1])

    target_mask_img = None
    if os.path.exists("{}.jpg".format(target_mask)):
        target_mask_img  = cv2.imread("{}.jpg".format(target_mask))
    elif os.path.exists("{}.png".format(target_mask)):
        target_mask_img  = cv2.imread("{}.png".format(target_mask))


    img = cv2.imread(im_p)
    img = resize_img(img)

    print("Input image")
    show_image(img)
    if target_mask_img is not None:
        print("Target mask")
        show_image(target_mask_img)

    if kwargs.get("gaussian"):
        img = image_gaussian(img, (3, 3))

    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if kwargs.get("pixel_range"):
        div = kwargs.get("pixel_range")
        if div == "kmeans":
            img_grey = kmeans_color_quantization(img_grey)
        elif div == "pix":
            img_grey = pixelate_bin(img_grey, 10, 100)
        else:
            img_grey = img_grey // div * div + div // 2
    else:
        th, img_grey = cv2.threshold(img_grey, 128, 255, cv2.THRESH_OTSU)

    p = int(kwargs.get("window_height") * kwargs.get("p"))
    mask = get_patch_degs(img_grey, p, kwargs.get("window_height"), kwargs.get("window_width"), kwargs.get("step_size_x"), kwargs.get("step_size_y"), kwargs.get("rotation"))

    return mask


# Run example
if __name__ == "__main__":
    kwargs =  {
        "window_height": 3, # the height of the window in pixels to use for the edge detection by making a pseudo-Boolean polynomial
        "window_width": 3, # the width of the window in pixels to use for the edge detection by making a pseudo-Boolean polynomial
        "step_size_y": 1, # how to step through the y axis across the image plane when extracting windows
        "step_size_x": 1, # how to step through the x axis across the image plane when extracting windows
        "p": 0.8, # threshold for binary classification of blob/edge regions. [0.1 - 0.9]
        "rotation": True, # if to check the patche's pBp degree in normal and transposed form
        "gaussian": True, # if to apply a gaussian blur to the image before edge detection
        "pixel_range": False, # if to apply a pixel range filter to the image before edge detection [kmeans to use a kmeans_color_quantization filter, pixelate_bin to pixalate the image, <number> to apply a threshold filter], otherwise use cv2.THRESH_OTSU
        "im_p": "examples/images/elephant.png", # image path
        "fine": False # if to use the fine-grained version of the algorithm TODO
    }

    start_time = time.perf_counter()
    mask = process_file(**kwargs)
    last = time.perf_counter() - start_time
    if mask is not None:
        print("PBp image edge mask")
        show_image(mask)
        print(pd.DataFrame([kwargs]).drop(["fine", "im_p"], axis=1))
        print(f"Time taken: {last:0.4f} seconds")
        print("="*80)
        print()


