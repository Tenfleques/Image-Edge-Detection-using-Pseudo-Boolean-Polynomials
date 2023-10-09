import sys
import os
import cv2
import tqdm
import numpy as np
import io
import plotly.io as pio
from PIL import Image
import math
import glob
import json

import logging 
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))

from pbp.reducer import pbp_instance, calculate_bin_size, BIT_ORDER
from image_utils.im_filters import group_ranges, pixelate_bin, kmeans_color_quantization, image_gaussian, resize_img

def json_default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


def similar_pbps(A, B):
    if not (A.shape == B.shape):
        return False

    A = np.cumsum(A, axis=-1)
    B = np.cumsum(B, axis=-1)

    diff = A - B
    return not (np.any(diff))


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

def patch_reducer(img, **kwargs):
    patch = kwargs.get("patch", [0, 20, 0, 20])
    rotation = kwargs.get("rotation", 0)
    shrink = kwargs.get("shrink", True)

    try:
        rotation = int(rotation)
    except Exception as err:
        logger.exception("making rotation variable integer {}".format(err))

    if isinstance(img, str):
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    h, w = img.shape[:2]

    pbp = {
        "coefficients": [],
        "variables": [],
        "error": "",
        "rank": 0
    }

    # print("[INFO] received patches of {}".format(patch))
    if isinstance(patch, str):
        patch = [int(i) for i in patch.split(",")]

    if len(patch) >= 4:
        # patch is x1, x2, y1, y2
        if patch[0] <= h and patch[1] <= h and patch[2] <= w and patch[1] <= w:
            patch = img[patch[0]: patch[1], patch[2]:patch[3]]
        else:
            pbp["error"] = "patch mismatch"
            return pbp
    else:
        pbp["error"] = "patch mismatch"
        return pbp

    patch = np.array(patch)
    rank = 0
    r_rank = 0
    try:
        k, y, b_size = pbp_instance(patch, shrink=shrink, return_byte_size=True)

        rank = getPBPDegree(y, b_size)
        pbp = {
            "coefficients": k,
            "variables": y,
            "error": "",
            "c": patch,
            "rank": rank
        }

    except Exception as er:
        logger.exception("gtting pbp and rank {}".format(er))
        try:
            if rotation:
                k_r, y_r, r_b_size = pbp_instance(patch.T, shrink=shrink, return_byte_size=True)
                r_rank = getPBPDegree(y_r, r_b_size)
                pbp["r_coefficients"] = k_r
                pbp["r_rank"] = r_rank
                pbp["r_variables"] = y_r

        except Exception as err:
            # traceback.print_exc()
            logger.exception("getting pbp instance {}".format(err))
            pbp = {
                "coefficients": [],
                "variables": [],
                "error": "{}".format(err),
                "rank": rank
            }

    return pbp


def get_patches(img, **kwargs):
    window_size = [int(kwargs.get("window_height", 12)), int(kwargs.get("window_width", 12))]
    step_size = [int(kwargs.get("step_size_y", 1)), int(kwargs.get("step_size_x", 1))]

    # print("[INFO] patch params {} {} ".format(window_size, step_size))

    if isinstance(img, str):
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    patches = []
    for i in tqdm.tqdm(range(0, h - window_size[0], step_size[0])):
        for j in range(0, w - window_size[1], step_size[1]):
            patches.append([i, i + window_size[0], j, j + window_size[1], i, j])

    return patches


def get_equivalent_instances(pbp_root: dict, pbps_compare: list, index: int):
    equivalent_instances = []
    pbp_root["coefficients"] = np.array(pbp_root["coefficients"])
    has_rotation = False

    if pbp_root.get("r_coefficients", None) is not None:
        has_rotation = True
        pbp_root["r_coefficients"] = np.array(pbp_root["r_coefficients"])

    for i, pbp_comp in enumerate(pbps_compare):
        if i == index:
            continue
        pbp_comp["coefficients"] = np.array(pbp_comp["coefficients"])

        if similar_pbps(pbp_comp["coefficients"], pbp_root["coefficients"]):
            equivalent_instances.append(i)
        else:
            # check for equivalence in rotated mode
            if has_rotation and pbp_comp.get("r_coefficients", None) is not None:
                pbp_comp["r_coefficients"] = np.array(pbp_comp["r_coefficients"])

                if similar_pbps(pbp_comp["r_coefficients"], pbp_root["coefficients"]):
                    equivalent_instances.append(i)
                else:
                    if similar_pbps(pbp_comp["r_coefficients"], pbp_root["r_coefficients"]):
                        equivalent_instances.append(i)

    if len(equivalent_instances):
        equivalent_instances.append(index)

    return equivalent_instances


def get_neighboring_patches(patch, patches):
    pass


def get_neighboring_inequivalent(pbp_root: dict, patches, pbps_compare: list):
    inequivalent_instances = []
    pbp_root["coefficients"] = np.array(pbp_root["coefficients"])

    for k, p in enumerate(patches):
        pbps_compare[k]["coefficients"] = np.array(pbps_compare[k]["coefficients"])

        if similar_pbps(pbps_compare[k]["coefficients"], pbp_root["coefficients"]):
            inequivalent_instances.append(k)

    return inequivalent_instances


def update_row_col(row, col):
    if col < 4:
        col += 1
    else:
        row += 1
        col = 1

    return row, col


def plot_results(img, degs, filename="", prep_images=[], target_mask=None):
    r = 1 + math.ceil(len(prep_images) / 2)
    
    titles = ["Input Image"]
    titles.extend([i["title"] for i in prep_images])
    titles.extend(["Degrees", "Degrees 3D"])

    specs = [[{'type': 'image'}, {'type': 'image'}] for i in range(r - 1)]
    specs.extend([[{'type': 'image'}, {'type': 'image'}]])

    if target_mask is not None:
        titles.extend(["Target Mask", "PBP Mask"])
        specs.extend([[{'type': 'image'}, {'type': 'image'}]])
        r += 1

    specs = np.array(specs).T
    specs[1, 1] = {'type': 'surface'}
    specs = specs.tolist()

    fig = make_subplots(
        rows=2, cols=r,
        subplot_titles=(titles),
        specs=specs
    )

    fig.add_trace(px.imshow(img).data[0], row=1, col=1)
    row = 1
    col = 2
    if len(prep_images):
        for prep_image in prep_images:
            fig.add_trace(px.imshow(prep_image["image"]).data[0], row=row, col=col)
            row, col = update_row_col(row, col)

    ranks_2d = px.imshow(degs, aspect=[1, 1], text_auto=True).data[0]

    fig.add_trace(ranks_2d, row=row, col=col)

    row, col = update_row_col(row, col)
    fig.add_trace(
        go.Surface(z=degs, colorscale='Viridis',
                    showscale=False, 
                    contours_z=dict(show=True, usecolormap=True,
                            highlightcolor="limegreen", project_z=True)), row=row, col=col)
    
    if target_mask is not None:
        row, col = update_row_col(row, col)
        target_mask_plot = px.imshow(target_mask).data[0]
        fig.add_trace(target_mask_plot, row=row, col=col)
        row, col = update_row_col(row, col)
        fig.add_trace(px.imshow(prep_images[-1]["image"]).data[0], row=row, col=col)
        fig.update_layout(height=800, width=1800, title_text="PBP edge detection process")
    else:
        fig.update_layout(height=1800, width=1400, title_text="PBP edge detection process")
        
    fig.update_layout(scene_camera_eye=dict(x=0.63, y=0, z=2.0))
    fig.update_layout(yaxis=dict(scaleanchor='x'))
    fig.update_layout(showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    # save degrees 
    np.save('{}.npy'.format(filename), degs)

    if filename:
        show_image(fig, filename)
    else:
        fig.show()


def show_image(fig, filepath):
    buf = io.BytesIO()
    pio.write_image(fig, buf)
    img = Image.open(buf)
    img.save(filepath)


def pbp_edge_detection(img_path="examples/im-seg/primitives.png", **kwargs):
    # prepare patches
    frame = cv2.imread(img_path)

    if frame is None:
        logger.error("Failed to read image {}".format(img_path))
        return None

    frame_g = frame.copy()

    prep_images = []
    if kwargs.get("gaussian"):
        frame_g = image_gaussian(frame, (3, 3))
        prep_images.append({
            "image": frame_g,
            "title": "Gaussian filter (3x3)"
        })
    else:
        prep_images.append({
            "image": frame_g,
            "title": "Gaussian filter - not implemented"
        })

    frame_thresh = frame_g.copy()
    gray_frame = cv2.cvtColor(frame_thresh, cv2.COLOR_BGR2GRAY)

    if kwargs.get("pixel_range"):
        # gray_frame = group_ranges(gray_frame, kwargs.get("pixel_range"))
        div = kwargs.get("pixel_range")
        if div == "kmeans":
            gray_frame = kmeans_color_quantization(gray_frame)
        elif div == "pix":
            gray_frame = pixelate_bin(gray_frame, 10, 100)
        else:
            gray_frame = gray_frame // div * div + div // 2
    else:
        th, gray_frame = cv2.threshold(gray_frame, 128, 255, cv2.THRESH_OTSU)

    prep_images.append({
        "image": gray_frame,
        "title": "Adaptive thresholding"
    })

    patches = get_patches(gray_frame, **kwargs)

    patches = np.array(patches)
    h, w = gray_frame.shape
    # get pbps
    pbps = []
    degs = np.zeros((h, w), dtype=int)
    patch_h = 0
    patch_w = 0

    for i in tqdm.tqdm(range(len(patches))):
        patch = patches[i]
        if patch_h == 0:
            patch_h = patch[1] - patch[0]
            patch_w = patch[3] - patch[2]

        pbp = patch_reducer(gray_frame, patch=patch, **kwargs)
        pbps.append(pbp)
        r = pbp["rank"]

        if pbp.get("r_rank", None) is not None:
            r = np.max([pbp["rank"], pbp["r_rank"]])

        degs[patch[4], patch[5]] = np.max([r, degs[patch[4], patch[5]]])

    if kwargs.get("p"):
        max_r = np.max(degs)
        cut_off_p = np.round(kwargs["p"] * max_r).astype(np.uint)

        print("[INFO] uncut degs {}".format(np.unique(degs)))
        cut_indices = np.nonzero(degs < cut_off_p)
        ranks_cut_off = degs.copy()
        ranks_cut_off[cut_indices] = 0
        print("[INFO] cut degs by {} of {} = {} -> {}".format(
            kwargs["p"], max_r, cut_off_p, np.unique(ranks_cut_off)))

        if not kwargs.get("fine", False):
            segment_mask = process_segments(ranks_cut_off, frame, patch_h, patch_w)
        else:
            segment_mask = process_segments_fine(ranks_cut_off, frame)

        prep_images.append({
            "image": segment_mask,
            "title": "Segment mask for p = {}".format(cut_off_p)
        })
    else:
        segment_mask = degs 
        prep_images.append({
            "image": degs,
            "title": "Segment mask (b4 post-processing)"
        })

    target_mask = None
    if kwargs.get("target_mask", None) is not None:
        try:
            if isinstance(kwargs["target_mask"], str):
                target_mask = cv2.imread(kwargs["target_mask"])
            else:
                target_mask = kwargs["target_mask"]
        except Exception as err:
            print("[ERROR] getting target segment {}".format(err))

    ranks_plot = np.flipud(degs)

    out_path = kwargs.get("out_path", "./out.png")
    plot_results(frame, ranks_plot, out_path, prep_images=prep_images, target_mask=target_mask)

    return segment_mask


def process_segments(degs, img, patch_h, patch_w):
    """
        process edge detection
    """
    img = img.copy()
    valued_indices = np.nonzero(degs)
    patch_h //= 2
    patch_w //= 2

    if len(img.shape) == 2:
        img[:, :] = 255
        for i in range(-patch_h, patch_h):
            img[valued_indices[0] + i, valued_indices[1]] = 0
        for i in range(-patch_w, patch_w):
            img[valued_indices[0], valued_indices[1] + i] = 0
    elif len(img.shape) == 3:
        img[:, :, :] = 255
        # img[:, :, 1] = 0
        for i in range(-patch_h, patch_h):
            img[valued_indices[0] + i, valued_indices[1], :] = 0
            # img[valued_indices[0] + i, valued_indices[1], 2] = 0

        for i in range(-patch_w, patch_w):
            img[valued_indices[0], valued_indices[1] + i, :] = 0
            # img[valued_indices[0], valued_indices[1] + i, :] = 0

        img[valued_indices[0], valued_indices[1], :] = 0
        # img[valued_indices[0], valued_indices[1], 2] = 0

    return img


def process_segments_fine(degs, img, flip=True):
    """
        process edge detection
    """
    print("processing fine edge detection")
    img = img.copy()
    valued_indices = np.nonzero(degs)

    bg = 0
    fg = 255

    if flip:
        bg = 255
        fg = 0

    if len(img.shape) == 2:
        img[:, :] = bg
        img[valued_indices[0], valued_indices[1]] = fg

    elif len(img.shape) == 3:
        img[:, :, :] = bg
        img[valued_indices[0], valued_indices[1], :] = fg

    return img


def process_file(im_p, **kwargs):
    print("[INFO] processing file {}".format(im_p))
    target_mask = ".".join(im_p.replace("images", "masks").split(".")[:-1])

    out_path_arr = im_p.split(os.sep)
    out_path_dir = os.path.join(*out_path_arr[:-2], "pbp-edges")
    os.makedirs(out_path_dir, exist_ok=True)

    out_path = out_path_arr[-1].split(".")[:-1]
    out_path = os.path.join(out_path_dir, "{}.png".format(".".join(out_path)))
    predicted_seg_mask_path = out_path.replace(".png", "-pbp-mask.png")

    if os.path.exists("{}.jpg".format(target_mask)):
        kwargs["target_mask"] = "{}.jpg".format(target_mask)
    elif os.path.exists("{}.png".format(target_mask)):
        kwargs["target_mask"] = "{}.png".format(target_mask)
    else:
        kwargs["target_mask"] = im_p
    
    print("[INFO] target mask {}".format(kwargs["target_mask"]))

    segment_mask = pbp_edge_detection(im_p, out_path=out_path, **kwargs)

    cv2.imwrite(predicted_seg_mask_path, segment_mask)


if __name__ == "__main__":
    with open("args.json") as fp:
        examples = json.load(fp)
        fp.close()

    for kwargs in examples:
        try:
            process_file(**kwargs)
        except Exception as err:
            logger.error("{}".format(err))
            logger.error("Processing {}".format(kwargs))