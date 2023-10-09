from numpy.lib import stride_tricks
import cv2
import os
import numpy as np
import base64


def get_patches(image, size=8):
    """
        non-overlapping patches of 
    """
    H, W = image.shape[:2]
    shape = [H // size, W // size] + [size, size]

    strides = [size * s for s in image.strides[:2]] + list(image.strides)

    patches = stride_tricks.as_strided(image, shape=shape, strides=strides,)

    return patches


def crop_and_pad_images(img_file, target_size=(360, 360, 3)):
    img_file_new = img_file.replace(".png", "-processed.png")
    if os.path.exists(img_file_new):
        return img_file_new

    image = cv2.imread(img_file)
    container = np.zeros((target_size), image.dtype)
    container[:image.shape[0], :image.shape[1], :] = image
    cv2.imwrite(img_file_new, container)

    return img_file_new


def resize_img(img_src, req_width=200):
    img = img_src.copy()
    if isinstance(img, str):
        img = cv2.imread(img)
    
    h, w, c = img.shape
    req_h = int(1.0 * req_width * h / w)

    img = cv2.resize(img, (req_width, req_h))

    return img


def frame_to_base_64(frame):
    if isinstance(frame, str):
        png_as_text = str(base64.b64encode(open(frame, "rb").read()))
    else:
        retval, buffer = cv2.imencode('.png', frame)
        png_as_text = str(base64.b64encode(buffer))

    png_as_text = "data:image/png;base64,{}".format(png_as_text[2:-1])
    return png_as_text


def image_to_xy(img):
    image_xy = np.nonzero(img)
    image_xy = np.array([image_xy[1][::-1], image_xy[0]])

    return image_xy


def get_spectral_patches(img, nrows=2, ncols=4):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    if isinstance(img, str):
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    h, w = img.shape
    org_h, org_w = img.shape

    if h % nrows != 0:
        h = int(np.ceil(h / nrows) * nrows)

    if w % ncols != 0:
        w = int(np.ceil(w / ncols) * ncols)

    container = np.zeros((h, w), dtype=int)
    container[:org_h, :org_w] = img

    container = container.reshape(h // nrows, nrows, -1, ncols).swapaxes(1, 2).reshape(-1, nrows, ncols)

    container = np.vsplit(container, nrows)
    return container


if __name__ == "__main__":
    img = np.array([
        np.arange(0, 10),
        np.arange(10, 20),
        np.arange(20, 30),
        np.arange(30, 40),
    ])
    print(img)
    print()

    nrows = 2
    ncols = 4

    segs = get_spectral_patches(img, nrows=nrows, ncols=ncols)

    for i in segs:
        print(i)
        print("="*20)
