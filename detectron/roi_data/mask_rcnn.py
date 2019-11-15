# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

"""Construct minibatches for Mask R-CNN training. Handles the minibatch blobs
that are specific to Mask R-CNN. Other blobs that are generic to RPN or
Fast/er R-CNN are handled by their respecitive roi_data modules.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
import cv2
from detectron.core.config import cfg
import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils
import detectron.utils.segms as segm_utils

logger = logging.getLogger(__name__)


def add_mask_rcnn_blobs(blobs, sampled_boxes, roidb, im_scale, batch_idx):
    """Add Mask R-CNN specific blobs to the input blob dictionary."""
    # Prepare the mask targets by associating one gt mask to each training roi
    # that has a fg (non-bg) class label.
    M = cfg.MRCNN.RESOLUTION

    input_w = roidb['input_width']
    input_h = roidb['input_height']
    polys_gt_inds = np.where(
        (roidb['gt_classes'] > 0) & (roidb['is_crowd'] == 0)
    )[0]
    polys_gt = [roidb['segms'][i] for i in polys_gt_inds]
    boxes_from_polys = segm_utils.polys_to_boxes(polys_gt)
    fg_inds = np.where(blobs['labels_int32'] > 0)[0]
    roi_has_mask = blobs['labels_int32'].copy()
    roi_has_mask[roi_has_mask > 0] = 1
    mask_fg_rois_per_this_image  = cfg.MRCNN.MAX_ROIS_PER_IM
    if fg_inds.shape[0] > 0:
        if fg_inds.size > mask_fg_rois_per_this_image:
            fg_inds = np.random.choice(
                fg_inds, size=mask_fg_rois_per_this_image, replace=False
            )
        # Class labels for the foreground rois
        mask_class_labels = blobs['labels_int32'][fg_inds]
        masks = blob_utils.zeros((fg_inds.shape[0], M**2), int32=True)
        all_person_masks = np.zeros((int(input_h/im_scale), int(input_w/im_scale)), dtype=np.float32)

        # Find overlap between all foreground rois and the bounding boxes
        # enclosing each segmentation
        rois_fg = sampled_boxes[fg_inds]
        overlaps_bbfg_bbpolys = box_utils.bbox_overlaps(
            rois_fg.astype(np.float32, copy=False),
            boxes_from_polys.astype(np.float32, copy=False)
        )
        # Map from each fg rois to the index of the mask with highest overlap
        # (measured by bbox overlap)
        fg_polys_inds = np.argmax(overlaps_bbfg_bbpolys, axis=1)

        # add fg targets
        for i in range(rois_fg.shape[0]):
            fg_polys_ind = fg_polys_inds[i]
            poly_gt = polys_gt[fg_polys_ind]
            roi_fg = rois_fg[i]
            # Rasterize the portion of the polygon mask within the given fg roi
            # to an M x M binary image
            mask = segm_utils.polys_to_mask_wrt_box(poly_gt, roi_fg, M)
            mask = np.array(mask > 0, dtype=np.int32)  # Ensure it's binary
            masks[i, :] = np.reshape(mask, M**2)
            # to an box_h x box_w binary image
            mask_wrt_bbox = segm_utils.convert_polys_to_mask_wrt_box(poly_gt, roi_fg)
            start_y, start_x = int(roi_fg[1]), int(roi_fg[0])
            end_y, end_x = start_y + mask_wrt_bbox.shape[0], start_x + mask_wrt_bbox.shape[1]
            all_person_masks[start_y:end_y, start_x:end_x] = mask_wrt_bbox
        proposal_all_mask = blob_utils.zeros((fg_inds.shape[0], M, M), int32=True)
        for i in range(rois_fg.shape[0]):
            roi_fg = rois_fg[i]
            w = roi_fg[2] - roi_fg[0]
            h = roi_fg[3] - roi_fg[1]
            w = int(np.maximum(w, 1))
            h = int(np.maximum(h, 1))
            proposal_mask = all_person_masks[int(roi_fg[1]):int(roi_fg[1])+h, int(roi_fg[0]):int(roi_fg[0])+w]
            # proposal_mask = proposal_mask.astype(np.float32)
            proposal_mask = cv2.resize(proposal_mask,(M,M))
            proposal_mask = (proposal_mask > 0.5).astype(np.int32)
            proposal_all_mask[i] = proposal_mask
    else:  # If there are no fg masks (it does happen)
        # The network cannot handle empty blobs, so we must provide a mask
        # We simply take the first bg roi, given it an all -1's mask (ignore
        # label), and label it with class zero (bg).
        bg_inds = np.where(blobs['labels_int32'] == 0)[0]
        # rois_fg is actually one background roi, but that's ok because ...
        rois_fg = sampled_boxes[bg_inds[0]].reshape((1, -1))
        # We give it an -1's blob (ignore label)
        masks = -blob_utils.ones((1, M**2), int32=True)
        # We label it with class = 0 (background)
        mask_class_labels = blob_utils.zeros((1, ))
        # Mark that the first roi has a mask
        roi_has_mask[0] = 1
        proposal_all_mask = -blob_utils.ones((1, M,M), int32=True)

    if cfg.MRCNN.CLS_SPECIFIC_MASK:
        masks = _expand_to_class_specific_mask_targets(masks, mask_class_labels)

    # Scale rois_fg and format as (batch_idx, x1, y1, x2, y2)
    rois_fg *= im_scale
    repeated_batch_idx = batch_idx * blob_utils.ones((rois_fg.shape[0], 1))
    rois_fg = np.hstack((repeated_batch_idx, rois_fg))

    # Update blobs dict with Mask R-CNN blobs
    blobs['mask_rois'] = rois_fg
    blobs['roi_has_mask_int32'] = roi_has_mask
    blobs['masks_int32'] = masks
#    blobs['mask_labels'] = np.argmax(masks.reshape((-1,cfg.MODEL.NUM_CLASSES,M,M)),axis=1).reshape((-1,M,M)).astype(np.int32)
#    blobs['mask_weights'] = np.ones(blobs['mask_labels'].shape, dtype=np.float32) 
    # add by wxh
    if cfg.MRCNN.USE_CLS_EMBS:
        fg_embs, bg_embs, fg_weights, bg_weights = masks_to_embs(masks.reshape((-1,cfg.MODEL.NUM_CLASSES,M,M)))
        # print('fg',fg_embs.max(), fg_embs.min())
        # print('bg',bg_embs.max(), bg_embs.min())
        fg_norms = np.sum(fg_embs,axis=(1,2))
        fg_norms[fg_norms!=0] =  28.* 28./ (fg_norms[fg_norms!=0]+1e-6)
        bg_norms = np.sum(bg_embs,axis=(1,2))
        bg_norms[bg_norms != 0] = 28.* 28. / (bg_norms[bg_norms != 0]+1e-6)

        blobs['fg_mask'] = np.repeat(np.reshape(fg_embs,(-1, 1, M, M)),2,axis=1)
        blobs['bg_mask'] = np.repeat(np.reshape(bg_embs, (-1, 1, M, M)),2,axis=1)
        blobs['fg_norm'] = np.repeat(np.reshape(fg_norms, (-1, 1)),2,axis=1)
        blobs['bg_norm'] = np.repeat(np.reshape(bg_norms, (-1, 1)),2,axis=1)

        blobs['mask_emb_fg_labels'] = np.ones((fg_embs.shape[0],1),dtype=np.int32)
        blobs['mask_emb_bg_labels'] = np.zeros((bg_embs.shape[0], 1),dtype=np.int32)
#        blobs['mask_emb_weights'] = np.vstack([fg_weights, bg_weights]).reshape((-1,1)).astype(np.float32)
    if cfg.MRCNN.BBOX_CASCADE_MASK_ON:
        blobs['inter_masks_int32']=proposal_all_mask




def _expand_to_class_specific_mask_targets(masks, mask_class_labels):
    """Expand masks from shape (#masks, M ** 2) to (#masks, #classes * M ** 2)
    to encode class specific mask targets.
    """
    assert masks.shape[0] == mask_class_labels.shape[0]
    M = cfg.MRCNN.RESOLUTION

    # Target values of -1 are "don't care" / ignore labels
    mask_targets = -blob_utils.ones(
        (masks.shape[0], cfg.MODEL.NUM_CLASSES * M**2), int32=True
    )

    for i in range(masks.shape[0]):
        cls = int(mask_class_labels[i])
        start = M**2 * cls
        end = start + M**2
        # Ignore background instance
        # (only happens when there is no fg samples in an image)
        if cls > 0:
            mask_targets[i, start:end] = masks[i, :]

    return mask_targets

def masks_to_embs(masks):
    # masks: batchsize, cls_num, M, M
    bs, cls_num = masks.shape[:2]
    fg_segs = []
    fg_weights = []
    bg_segs = []
    bg_weights = []

    for i in range(bs):
        # print('mask 0', masks[i,0].max(), masks[i,0].min())
        # print('mask 1', masks[i, 1].max(), masks[i, 1].min())
        fg_mask = (masks[i,1] > 0).astype(np.float32)
        bg_mask = (fg_mask == 0).astype(np.float32)
        # print('fg mask',fg_mask.max(),fg_mask.min())
        # print('bg mask',bg_mask.max(),bg_mask.min())
        fg_segs.append(fg_mask)
        bg_segs.append(bg_mask)
        if fg_mask.max()==1:
            fg_weights.append(1)
        else:
            fg_weights.append(0)
        if bg_mask.max()==1:
            bg_weights.append(1)
        else:
            bg_weights.append(0)
    return np.asarray(fg_segs), np.asarray(bg_segs), np.asarray(fg_weights), np.asarray(bg_weights)



