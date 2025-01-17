# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

"""Keypoint utilities (somewhat specific to COCO keypoints)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np

from detectron.core.config import cfg
import detectron.utils.blob as blob_utils


def get_keypoints():
    """Get the COCO keypoints and their left/right flip coorespondence map."""
    # Keypoints are not available in the COCO json for the test split, so we
    # provide them here.
    keypoints = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]
    keypoint_flip_map = {
        'left_eye': 'right_eye',
        'left_ear': 'right_ear',
        'left_shoulder': 'right_shoulder',
        'left_elbow': 'right_elbow',
        'left_wrist': 'right_wrist',
        'left_hip': 'right_hip',
        'left_knee': 'right_knee',
        'left_ankle': 'right_ankle'
    }
    return keypoints, keypoint_flip_map


def get_person_class_index():
    """Index of the person class in COCO."""
    return 1


def flip_keypoints(keypoints, keypoint_flip_map, keypoint_coords, width):
    """Left/right flip keypoint_coords. keypoints and keypoint_flip_map are
    accessible from get_keypoints().
    """
    flipped_kps = keypoint_coords.copy()
    for lkp, rkp in keypoint_flip_map.items():
        lid = keypoints.index(lkp)
        rid = keypoints.index(rkp)
        flipped_kps[:, :, lid] = keypoint_coords[:, :, rid]
        flipped_kps[:, :, rid] = keypoint_coords[:, :, lid]

    # Flip x coordinates
    flipped_kps[:, 0, :] = width - flipped_kps[:, 0, :] - 1
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = np.where(flipped_kps[:, 2, :] == 0)
    flipped_kps[inds[0], 0, inds[1]] = 0
    return flipped_kps


def flip_heatmaps(heatmaps):
    """Flip heatmaps horizontally."""
    keypoints, flip_map = get_keypoints()
    heatmaps_flipped = heatmaps.copy()
    for lkp, rkp in flip_map.items():
        lid = keypoints.index(lkp)
        rid = keypoints.index(rkp)
        heatmaps_flipped[:, rid, :, :] = heatmaps[:, lid, :, :]
        heatmaps_flipped[:, lid, :, :] = heatmaps[:, rid, :, :]
    heatmaps_flipped = heatmaps_flipped[:, :, :, ::-1]
    return heatmaps_flipped


def heatmaps_to_keypoints(maps, rois):
    """Extract predicted keypoint locations from heatmaps. Output has shape
    (#rois, 4, #keypoints) with the 4 rows corresponding to (x, y, logit, prob)
    for each keypoint.
    """
    # This function converts a discrete image coordinate in a HEATMAP_SIZE x
    # HEATMAP_SIZE image to a continuous keypoint coordinate. We maintain
    # consistency with keypoints_to_heatmap_labels by using the conversion from
    # Heckbert 1990: c = d + 0.5, where d is a discrete coordinate and c is a
    # continuous coordinate.
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]

    widths = rois[:, 2] - rois[:, 0]
    heights = rois[:, 3] - rois[:, 1]
    widths = np.maximum(widths, 1)
    heights = np.maximum(heights, 1)
    widths_ceil = np.ceil(widths)
    heights_ceil = np.ceil(heights)

    # NCHW to NHWC for use with OpenCV
    maps = np.transpose(maps, [0, 2, 3, 1])
    min_size = cfg.KRCNN.INFERENCE_MIN_SIZE
    xy_preds = np.zeros(
        (len(rois), 4, cfg.KRCNN.NUM_KEYPOINTS), dtype=np.float32)
    for i in range(len(rois)):
        if min_size > 0:
            roi_map_width = int(np.maximum(widths_ceil[i], min_size))
            roi_map_height = int(np.maximum(heights_ceil[i], min_size))
        else:
            roi_map_width = widths_ceil[i]
            roi_map_height = heights_ceil[i]
        width_correction = widths[i] / roi_map_width
        height_correction = heights[i] / roi_map_height
        roi_map = cv2.resize(
            maps[i], (roi_map_width, roi_map_height),
            interpolation=cv2.INTER_CUBIC)
        # Bring back to CHW
        roi_map = np.transpose(roi_map, [2, 0, 1])
        roi_map_probs = scores_to_probs(roi_map.copy())
        w = roi_map.shape[2]
        for k in range(cfg.KRCNN.NUM_KEYPOINTS):
            pos = roi_map[k, :, :].argmax()
            x_int = pos % w
            y_int = (pos - x_int) // w
            assert (roi_map_probs[k, y_int, x_int] ==
                    roi_map_probs[k, :, :].max())
            x = (x_int + 0.5) * width_correction
            y = (y_int + 0.5) * height_correction
            xy_preds[i, 0, k] = x + offset_x[i]
            xy_preds[i, 1, k] = y + offset_y[i]
            xy_preds[i, 2, k] = roi_map[k, y_int, x_int]
            xy_preds[i, 3, k] = roi_map_probs[k, y_int, x_int]

    return xy_preds


def keypoints_to_heatmap_labels(keypoints, rois):
    """Encode keypoint location in the target heatmap for use in
    SoftmaxWithLoss.
    """
    # Maps keypoints from the half-open interval [x1, x2) on continuous image
    # coordinates to the closed interval [0, HEATMAP_SIZE - 1] on discrete image
    # coordinates. We use the continuous <-> discrete conversion from Heckbert
    # 1990 ("What is the coordinate of a pixel?"): d = floor(c) and c = d + 0.5,
    # where d is a discrete coordinate and c is a continuous coordinate.
    assert keypoints.shape[2] == cfg.KRCNN.NUM_KEYPOINTS

    shape = (len(rois), cfg.KRCNN.NUM_KEYPOINTS)
    heatmaps = blob_utils.zeros(shape)
    weights = blob_utils.zeros(shape)

    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = cfg.KRCNN.HEATMAP_SIZE / (rois[:, 2] - rois[:, 0])
    scale_y = cfg.KRCNN.HEATMAP_SIZE / (rois[:, 3] - rois[:, 1])

    for kp in range(keypoints.shape[2]):
        vis = keypoints[:, 2, kp] > 0
        x = keypoints[:, 0, kp].astype(np.float32)
        y = keypoints[:, 1, kp].astype(np.float32)
        # Since we use floor below, if a keypoint is exactly on the roi's right
        # or bottom boundary, we shift it in by eps (conceptually) to keep it in
        # the ground truth heatmap.
        x_boundary_inds = np.where(x == rois[:, 2])[0]
        y_boundary_inds = np.where(y == rois[:, 3])[0]
        x = (x - offset_x) * scale_x
        x = np.floor(x)
        if len(x_boundary_inds) > 0:
            x[x_boundary_inds] = cfg.KRCNN.HEATMAP_SIZE - 1

        y = (y - offset_y) * scale_y
        y = np.floor(y)
        if len(y_boundary_inds) > 0:
            y[y_boundary_inds] = cfg.KRCNN.HEATMAP_SIZE - 1

        valid_loc = np.logical_and(
            np.logical_and(x >= 0, y >= 0),
            np.logical_and(
                x < cfg.KRCNN.HEATMAP_SIZE, y < cfg.KRCNN.HEATMAP_SIZE))

        valid = np.logical_and(valid_loc, vis)
        valid = valid.astype(np.int32)

        lin_ind = y * cfg.KRCNN.HEATMAP_SIZE + x
        heatmaps[:, kp] = lin_ind * valid
        weights[:, kp] = valid

    return heatmaps, weights

def keypoints_to_global_heatmap_labels(keypoints, rois, input_h, input_w):

    assert keypoints.shape[2] == cfg.KRCNN.NUM_KEYPOINTS
    M = cfg.KRCNN.HEATMAP_SIZE
    shape = (len(rois), cfg.KRCNN.NUM_KEYPOINTS)
    heatmaps = blob_utils.zeros(shape)
    weights = blob_utils.zeros(shape)
    global_heatmaps = np.zeros((input_h, input_w, cfg.KRCNN.NUM_KEYPOINTS),dtype=np.float32)
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = cfg.KRCNN.HEATMAP_SIZE / (rois[:, 2] - rois[:, 0])
    scale_y = cfg.KRCNN.HEATMAP_SIZE / (rois[:, 3] - rois[:, 1])

    for kp in range(keypoints.shape[2]):
        vis = keypoints[:, 2, kp] > 0
        x = keypoints[:, 0, kp].astype(np.float32)
        y = keypoints[:, 1, kp].astype(np.float32)
        # valid_loc = np.logical_and(x<input_w,y<input_h)
        # vis = np.logical_and(vis, valid_loc)
        # valid_x = x[vis].astype(np.int32)
        # valid_y = y[vis].astype(np.int32)
        # for i_loc in range(valid_x.shape[0]):
        #     mu_x = valid_x[i_loc]
        #     mu_y = valid_y[i_loc]
        #     range_x = np.arange(mu_x - 1, mu_x + 2)
        #     range_y = np.arange(mu_y - 1, mu_y + 2)
        #     range_x[range_x<0] = 0
        #     range_x[range_x>=input_w]=input_w-1
        #     range_y[range_y<1] = 0
        #     range_y[range_y>=input_h]=input_h-1
        #     global_heatmaps[range_y, range_x, kp] = 1
        # Since we use floor below, if a keypoint is exactly on the roi's right
        # or bottom boundary, we shift it in by eps (conceptually) to keep it in
        # the ground truth heatmap.
        x_boundary_inds = np.where(x == rois[:, 2])[0]
        y_boundary_inds = np.where(y == rois[:, 3])[0]
        x = (x - offset_x) * scale_x
        x = np.floor(x)
        if len(x_boundary_inds) > 0:
            x[x_boundary_inds] = cfg.KRCNN.HEATMAP_SIZE - 1

        y = (y - offset_y) * scale_y
        y = np.floor(y)
        if len(y_boundary_inds) > 0:
            y[y_boundary_inds] = cfg.KRCNN.HEATMAP_SIZE - 1

        valid_loc = np.logical_and(
            np.logical_and(x >= 0, y >= 0),
            np.logical_and(
                x < cfg.KRCNN.HEATMAP_SIZE, y < cfg.KRCNN.HEATMAP_SIZE))

        valid = np.logical_and(valid_loc, vis)
        valid_x = x[valid] / scale_x[valid] + offset_x[valid]
        valid_x = valid_x.astype(np.int32)
        valid_y = y[valid] / scale_y[valid] + offset_y[valid]
        valid_y = valid_y.astype(np.int32)
        global_heatmaps[valid_y, valid_x, kp] = 1

        valid = valid.astype(np.int32)

        lin_ind = y * cfg.KRCNN.HEATMAP_SIZE + x
        heatmaps[:, kp] = lin_ind * valid
        weights[:, kp] = valid
    proposal_all_heatmaps = blob_utils.zeros((rois.shape[0], cfg.KRCNN.NUM_KEYPOINTS, M, M), int32=True)
    for i in range(rois.shape[0]):
        roi_fg = rois[i]
        w = roi_fg[2] - roi_fg[0]
        h = roi_fg[3] - roi_fg[1]
        w = int(np.maximum(w, 1))
        h = int(np.maximum(h, 1))
        proposal_hm = global_heatmaps[int(roi_fg[1]):int(roi_fg[1]) + h, int(roi_fg[0]):int(roi_fg[0]) + w, :]
        proposal_hm = cv2.resize(proposal_hm, (M, M))
        proposal_hm = (proposal_hm > 0.5).astype(np.int32)
        proposal_all_heatmaps[i] = proposal_hm.transpose((2,0,1))

    return heatmaps, weights, proposal_all_heatmaps

def scores_to_probs(scores):
    """Transforms CxHxW of scores to probabilities spatially."""
    channels = scores.shape[0]
    for c in range(channels):
        temp = scores[c, :, :]
        max_score = temp.max()
        temp = np.exp(temp - max_score) / np.sum(np.exp(temp - max_score))
        scores[c, :, :] = temp
    return scores




def nms_oks(kp_predictions, rois, thresh):
    """Nms based on kp predictions."""
    scores = np.mean(kp_predictions[:, 2, :], axis=1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = compute_oks(
            kp_predictions[i], rois[i], kp_predictions[order[1:]],
            rois[order[1:]])
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def compute_oks(src_keypoints, src_roi, dst_keypoints, dst_roi):
    """Compute OKS for predicted keypoints wrt gt_keypoints.
    src_keypoints: 4xK
    src_roi: 4x1
    dst_keypoints: Nx4xK
    dst_roi: Nx4
    """

    sigmas = np.array([
        .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87,
        .87, .89, .89]) / 10.0
    vars = (sigmas * 2)**2

    # area
    src_area = (src_roi[2] - src_roi[0] + 1) * (src_roi[3] - src_roi[1] + 1)

    # measure the per-keypoint distance if keypoints visible
    dx = dst_keypoints[:, 0, :] - src_keypoints[0, :]
    dy = dst_keypoints[:, 1, :] - src_keypoints[1, :]

    e = (dx**2 + dy**2) / vars / (src_area + np.spacing(1)) / 2
    e = np.sum(np.exp(-e), axis=1) / e.shape[1]

    return e
