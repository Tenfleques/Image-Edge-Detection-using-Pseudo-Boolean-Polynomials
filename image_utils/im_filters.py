import numpy as np
import cv2
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from patches import resize_img


def gabor(sigma, theta, Lambda, psi, gamma):
    """
        In image processing, a Gabor filter, named after Dennis Gabor, is a linear filter used for texture analysis, which essentially means that it analyzes whether there is any specific frequency content in the image in specific directions in a localized region around the point or region of analysis. Frequency and orientation representations of Gabor filters are claimed by many contemporary vision scientists to be similar to those of the human visual system.[1] They have been found to be particularly appropriate for texture representation and discrimination. In the spatial domain, a 2-D Gabor filter is a Gaussian kernel function modulated by a sinusoidal plane wave
    """
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3  # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2))
    gb = gb * np.cos(2 * np.pi / Lambda * x_theta + psi)

    return gb


def group_ranges(img_src, range_length):
    img = img_src.copy()
    num_groups = 255 // range_length

    groups = np.array_split(np.arange(0, 256), num_groups)
    grp_extremes = [[np.min(g), np.max(g)] for g in groups]

    for i, grp in enumerate(grp_extremes):
        indices = np.nonzero(np.logical_and(grp[0] <= img, img < grp[1]))
        img[indices] = grp[1]

    img = img.astype(np.uint8)

    return img


def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h * w], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    eps_ctrit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001)

    init_c = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(samples, clusters, None, eps_ctrit, rounds, init_c)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]

    return res.reshape((image.shape))


def pixelate_bin(img_src, window, threshold):
    img = img_src.copy()
    n, m = img.shape
    n, m = n - n % window, m - m % window

    img1 = np.zeros((n, m))
    if np.max(img) > 1.0:
        img = img / 255

    for x in range(0, n, window):
        for y in range(0, m, window):
            # print(img[x: x + window, y: y + window].mean())
            if img[x: x + window, y: y + window].mean() > threshold:
                img1[x: x + window, y: y + window] = 1

    return img1


def pixelate_rgb(img_src, window):
    img = img_src.copy().astype(float)
    n, m, c = img.shape
    n, m = n - n % window, m - m % window
    img1 = np.zeros((n, m, c))

    if np.max(img) > 1.0:
        img /= 255.

    for x in range(0, n, window):
        for y in range(0, m, window):
            img1[x:x + window, y: y + window] = img[x:x + window, y: y + window].mean(axis=(0, 1))

    return img1


def image_gaussian(img_src, kernel=(1, 1)):
    img = img_src.copy()
    norm_img = np.zeros((img.shape[0], img.shape[1]))
    if np.max(img) > 1.0:
        img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)

    img = cv2.GaussianBlur(img, kernel, 0)
    img = (img * 255).astype(np.uint8)

    return img

