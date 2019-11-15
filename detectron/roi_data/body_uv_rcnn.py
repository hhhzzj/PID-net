# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
#
from scipy.io import loadmat
import copy
import cv2
import logging
import numpy as np
import random
#

from detectron.core.config import cfg
import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils
import detectron.utils.segms as segm_utils
import detectron.utils.densepose_methods as dp_utils

#
from memory_profiler import profile
#
import os
#
logger = logging.getLogger(__name__)
#
DP = dp_utils.DensePoseMethods()
#

def add_body_uv_rcnn_blobs(blobs, sampled_boxes, roidb, im_scale, batch_idx):
    IsFlipped = roidb['flipped']
    M = cfg.BODY_UV_RCNN.HEATMAP_SIZE
    #
    polys_gt_inds = np.where(roidb['ignore_UV_body'] == 0)[0]
    boxes_from_polys = [roidb['boxes'][i,:] for i in polys_gt_inds]
    input_w = roidb['input_width']
    input_h = roidb['input_height']
    if not(boxes_from_polys):
        pass
    else:
        boxes_from_polys = np.vstack(boxes_from_polys)
    boxes_from_polys = np.array(boxes_from_polys)

    fg_inds = np.where(blobs['labels_int32'] > 0)[0]
    roi_has_mask = np.zeros( blobs['labels_int32'].shape )

    if (bool(boxes_from_polys.any()) & (fg_inds.shape[0] > 0) ):
        rois_fg = sampled_boxes[fg_inds]
        #
        rois_fg.astype(np.float32, copy=False)
        boxes_from_polys.astype(np.float32, copy=False)
        #
        overlaps_bbfg_bbpolys = box_utils.bbox_overlaps(
            rois_fg.astype(np.float32, copy=False),
            boxes_from_polys.astype(np.float32, copy=False))
        fg_polys_value = np.max(overlaps_bbfg_bbpolys, axis=1)
        fg_inds = fg_inds[fg_polys_value>0.7]
    all_person_masks = np.zeros((int(input_h), int(input_w)), dtype=np.float32)
    if (bool(boxes_from_polys.any()) & (fg_inds.shape[0] > 0) ):
        # controle the number of roi
        if fg_inds.shape[0]>6:
           fg_inds = fg_inds[:6]
        for jj in fg_inds:
            roi_has_mask[jj] = 1
         
        # Create blobs for densepose supervision.
        ################################################## The mask
        All_labels = blob_utils.zeros((fg_inds.shape[0], M ** 2), int32=True)
        All_Weights = blob_utils.zeros((fg_inds.shape[0], M ** 2), int32=True)
        ################################################# The points
        X_points = blob_utils.zeros((fg_inds.shape[0], 196), int32=False)
        Y_points = blob_utils.zeros((fg_inds.shape[0], 196), int32=False)
        Ind_points = blob_utils.zeros((fg_inds.shape[0], 196), int32=True)
        I_points = blob_utils.zeros((fg_inds.shape[0], 196), int32=True)
        U_points = blob_utils.zeros((fg_inds.shape[0], 196), int32=False)
        V_points = blob_utils.zeros((fg_inds.shape[0], 196), int32=False)
        Uv_point_weights = blob_utils.zeros((fg_inds.shape[0], 196), int32=False)
        #################################################

        rois_fg = sampled_boxes[fg_inds]
        overlaps_bbfg_bbpolys = box_utils.bbox_overlaps(
            rois_fg.astype(np.float32, copy=False),
            boxes_from_polys.astype(np.float32, copy=False))
        fg_polys_inds = np.argmax(overlaps_bbfg_bbpolys, axis=1)


        rois = np.copy(rois_fg)
        for i in range(rois_fg.shape[0]):
            #
            fg_polys_ind = polys_gt_inds[ fg_polys_inds[i] ]
            #
            Ilabel = segm_utils.GetDensePoseMask( roidb['dp_masks'][ fg_polys_ind ] )
            #
            GT_I = np.array(roidb['dp_I'][ fg_polys_ind ])
            GT_U = np.array(roidb['dp_U'][ fg_polys_ind ])
            GT_V = np.array(roidb['dp_V'][ fg_polys_ind ])
            GT_x = np.array(roidb['dp_x'][ fg_polys_ind ])
            GT_y = np.array(roidb['dp_y'][ fg_polys_ind ])
            GT_weights = np.ones(GT_I.shape).astype(np.float32)
            #
            ## Do the flipping of the densepose annotation !
            if(IsFlipped):
                GT_I,GT_U,GT_V,GT_x,GT_y,Ilabel = DP.get_symmetric_densepose(GT_I,GT_U,GT_V,GT_x,GT_y,Ilabel)
            #
            roi_fg = rois_fg[i]
            roi_gt = boxes_from_polys[fg_polys_inds[i],:]
            #
            x1 = roi_fg[0]  ;   x2 = roi_fg[2]
            y1 = roi_fg[1]  ;   y2 = roi_fg[3]
            #
            x1_source = roi_gt[0];  x2_source = roi_gt[2]
            y1_source = roi_gt[1];  y2_source = roi_gt[3]
            #
            x_targets  = ( np.arange(x1,x2, (x2 - x1)/M ) - x1_source ) * ( 256. / (x2_source-x1_source) )  
            y_targets  = ( np.arange(y1,y2, (y2 - y1)/M ) - y1_source ) * ( 256. / (y2_source-y1_source) )  
            #
            x_targets = x_targets[0:M] ## Strangely sometimes it can be M+1, so make sure size is OK!
            y_targets = y_targets[0:M]
            #
            [X_targets,Y_targets] = np.meshgrid( x_targets, y_targets )
            New_Index = cv2.remap(Ilabel,X_targets.astype(np.float32), Y_targets.astype(np.float32), interpolation=cv2.INTER_NEAREST, borderMode= cv2.BORDER_CONSTANT, borderValue=(0))
            #
            All_L = np.zeros(New_Index.shape)
            All_W = np.ones(New_Index.shape)
            #
            All_L = New_Index
            #
            gt_length_x = x2_source - x1_source
            gt_length_y = y2_source - y1_source
            #
            GT_y =  ((  GT_y / 256. * gt_length_y  ) + y1_source - y1 ) *  ( M /  ( y2 - y1 ) )
            GT_x =  ((  GT_x / 256. * gt_length_x  ) + x1_source - x1 ) *  ( M /  ( x2 - x1 ) )
            #
            GT_I[GT_y<0] = 0
            GT_I[GT_y>(M-1)] = 0
            GT_I[GT_x<0] = 0
            GT_I[GT_x>(M-1)] = 0
            #
            points_inside = GT_I>0
            GT_U = GT_U[points_inside]
            GT_V = GT_V[points_inside]
            GT_x = GT_x[points_inside]
            GT_y = GT_y[points_inside]
            GT_weights = GT_weights[points_inside]
            GT_I = GT_I[points_inside]


            #
            X_points[i, 0:len(GT_x)] = GT_x
            Y_points[i, 0:len(GT_y)] = GT_y
            Ind_points[i, 0:len(GT_I)] = i
            I_points[i, 0:len(GT_I)] = GT_I
            U_points[i, 0:len(GT_U)] = GT_U
            V_points[i, 0:len(GT_V)] = GT_V
            Uv_point_weights[i, 0:len(GT_weights)] = GT_weights
            #
            All_labels[i, :] = np.reshape(All_L.astype(np.int32), M ** 2)
            All_Weights[i, :] = np.reshape(All_W.astype(np.int32), M ** 2)
            ##
            # proposal based segmentation
            p_mask = (Ilabel>0).astype(np.float32)
            target_roi = roi_gt * im_scale
            p_mask = cv2.resize(p_mask,(int(target_roi[2]-target_roi[0]), int(target_roi[3]-target_roi[1])))
            p_mask = (p_mask>0.5).astype(np.float32)
            start_y, start_x = int(target_roi[1]), int(target_roi[0])
            end_y, end_x = start_y + p_mask.shape[0], start_x + p_mask.shape[1]
            # if all_person_masks[start_y:end_y, start_x:end_x].shape[0]!=p_mask.shape[0] or all_person_masks[start_y:end_y, start_x:end_x].shape[1]!=p_mask.shape[1]:
            #     print('shape exception:',all_person_masks[start_y:end_y, start_x:end_x].shape,p_mask.shape)
            #     print('roi:',target_roi)
            #     print(start_y,end_y, start_x,end_x)
            #     print('input image:',all_person_masks.shape)
            #     assert False
            all_person_masks[start_y:end_y, start_x:end_x]=p_mask
    else:
        bg_inds = np.where(blobs['labels_int32'] == 0)[0]
        #
        if(len(bg_inds)==0):
            rois_fg = sampled_boxes[0].reshape((1, -1))
        else:
            rois_fg = sampled_boxes[bg_inds[0]].reshape((1, -1))

        roi_has_mask[0] = 1
        #
        X_points = blob_utils.zeros((1, 196), int32=False)
        Y_points = blob_utils.zeros((1, 196), int32=False)
        Ind_points = blob_utils.zeros((1, 196), int32=True)
        I_points = blob_utils.zeros((1,196), int32=True)
        U_points = blob_utils.zeros((1, 196), int32=False)
        V_points = blob_utils.zeros((1, 196), int32=False)
        Uv_point_weights = blob_utils.zeros((1, 196), int32=False)
        #
        All_labels = -blob_utils.ones((1, M ** 2), int32=True) * 0 ## zeros
        All_Weights = -blob_utils.ones((1, M ** 2), int32=True) * 0 ## zeros
    #
    rois_fg *= im_scale
    repeated_batch_idx = batch_idx * blob_utils.ones((rois_fg.shape[0], 1))
    rois_fg = np.hstack((repeated_batch_idx, rois_fg))
    #
    K = cfg.BODY_UV_RCNN.NUM_PATCHES
    #
    u_points = np.copy(U_points)
    v_points = np.copy(V_points)
    U_points = np.tile( U_points , [1,K+1] )
    V_points = np.tile( V_points , [1,K+1] )
    Uv_Weight_Points = np.zeros(U_points.shape)
    #
    for jjj in xrange(1,K+1):
        Uv_Weight_Points[ : , jjj * I_points.shape[1]  : (jjj+1) * I_points.shape[1] ] = ( I_points == jjj ).astype(np.float32)
    #

    # person masks here
    person_mask = (All_labels>0).astype(np.int32)
    # extra
    # index_targets = np.zeros_like(person_mask).reshape((-1,M,M)).astype(np.int32)
    # index_targets_weights = np.zeros_like(index_targets)
    # u_targets = np.zeros((index_targets.shape[0],25,M,M),dtype=np.float32)
    # v_targets = np.zeros((index_targets.shape[0], 25, M, M),dtype=np.float32)
    # uv_weights = np.zeros((index_targets.shape[0], 25, M, M),dtype=np.float32)
    # for ibatch in range(index_targets.shape[0]):
    #     for i_surface in range(1,K+1):
    #         points_i = I_points[ibatch] == i_surface
    #         if len(points_i)>0:
    #             points_x = np.asarray(X_points[ibatch][points_i], dtype=np.int32).reshape((-1,1))
    #             points_y = np.asarray(Y_points[ibatch][points_i], dtype=np.int32).reshape((-1,1))
    #             points_u = u_points[ibatch][points_i].reshape((1, -1))
    #             points_v = v_points[ibatch][points_i].reshape((1, -1))
    #             locs = np.hstack([points_x, points_y])
    #
    #             for step in [1]:
    #                 x_plus_locs = np.copy(points_x) + step
    #                 y_plus_locs = np.copy(points_y) + step
    #                 x_minus_locs = np.copy(points_x) - step
    #                 y_minus_locs = np.copy(points_y) - step
    #
    #                 locs = np.vstack([locs, np.hstack([x_plus_locs, y_plus_locs])])
    #                 locs = np.vstack([locs, np.hstack([x_plus_locs, y_minus_locs])])
    #                 locs = np.vstack([locs, np.hstack([x_minus_locs, y_plus_locs])])
    #                 locs = np.vstack([locs, np.hstack([x_minus_locs, y_minus_locs])])
    #
    #             locs[locs < 0] = 0.
    #             locs[locs >= M] = M - 1
    #
    #             points_u = np.repeat(points_u, 5, axis=0).reshape((-1))
    #             points_v = np.repeat(points_v, 5, axis=0).reshape((-1))
    #
    #
    #             index_targets[ibatch][locs[:,1], locs[:, 0]] = i_surface
    #             index_targets_weights[ibatch][locs[:, 1], locs[:, 0]] = 1
    #             u_targets[ibatch, i_surface][locs[:, 1], locs[:, 0]] = points_u
    #             v_targets[ibatch, i_surface][locs[:, 1], locs[:, 0]] = points_v
    #             uv_weights[ibatch, i_surface][locs[:, 1], locs[:, 0]] = 1.
    #     if random.random() <= 0.5:
    #         _,index_targets[ibatch], v_targets[ibatch], v_targets[ibatch], index_targets_weights[ibatch], uv_weights[ibatch] = expand_dp_targets(All_labels[ibatch].reshape((M,M)),
    #                                                                                                                                              index_targets[ibatch], v_targets[ibatch],
    #                                                                                                                                              v_targets[ibatch],
    #                                                                                                                                              index_targets_weights[ibatch],
    #                                                                                                                                              uv_weights[ibatch])

    # proposal all masks here

    if (bool(boxes_from_polys.any()) & (fg_inds.shape[0] > 0)):
        proposal_all_mask = blob_utils.zeros((fg_inds.shape[0], M,M), int32=True)
        for i in range(rois_fg.shape[0]):

            roi_fg = rois_fg[i][1:]

            proposal_mask = all_person_masks[int(roi_fg[1]):int(roi_fg[3]), int(roi_fg[0]):int(roi_fg[2])]
            proposal_mask = cv2.resize(proposal_mask,(M,M))
            proposal_mask = (proposal_mask>0.5).astype(np.int32)
            proposal_all_mask[i] = proposal_mask
    else:
        proposal_all_mask = -blob_utils.ones((1, M,M), int32=True) * 0  ## zeros

    ################
    # Update blobs dict with Mask R-CNN blobs
    ###############
    #
    blobs['body_mask_labels'] = person_mask.reshape((-1,M,M))
    blobs['body_uv_rois'] = np.array(rois_fg)
    blobs['roi_has_body_uv_int32'] = np.array(roi_has_mask).astype(np.int32)
    ##
    blobs['body_uv_ann_labels'] = np.array(All_labels).astype(np.int32)
    blobs['body_uv_ann_weights'] = np.array(All_Weights).astype(np.float32)
    #
    ##########################
    blobs['body_uv_X_points'] = X_points.astype(np.float32)
    blobs['body_uv_Y_points'] = Y_points.astype(np.float32)
    blobs['body_uv_Ind_points'] = Ind_points.astype(np.float32)
    blobs['body_uv_I_points'] = I_points.astype(np.float32)
    blobs['body_uv_U_points'] = U_points.astype(np.float32)  #### VERY IMPORTANT :   These are switched here :
    blobs['body_uv_V_points'] = V_points.astype(np.float32)
    blobs['body_uv_point_weights'] = Uv_Weight_Points.astype(np.float32)
    ###################
    # extra
    # blobs['body_uv_Index_targets'] = index_targets
    # blobs['body_uv_Index_targets_weights'] = index_targets_weights.astype(np.float32)
    # blobs['body_uv_U_targets'] = u_targets
    # blobs['body_uv_V_targets'] = v_targets
    # blobs['body_uv_weights'] = uv_weights
    ################
    # add by wxh
    if cfg.BODY_UV_RCNN.USE_CLS_EMBS:
        fg_embs, bg_embs, fg_weights, bg_weights = masks_to_embs(All_labels.reshape((-1,M,M)))
        # print('fg',fg_embs.max(), fg_embs.min())
        # print('bg',bg_embs.max(), bg_embs.min())
        fg_norms = np.sum(fg_embs,axis=(1,2))
        fg_norms[fg_norms!=0] =  56.*56./ fg_norms[fg_norms!=0]
        bg_norms = np.sum(bg_embs,axis=(1,2))
        bg_norms[bg_norms != 0] = 56.*56. / bg_norms[bg_norms != 0]

        blobs['fg_mask'] = np.repeat(np.reshape(fg_embs,(-1, 1, M, M)),2,axis=1)
        blobs['bg_mask'] = np.repeat(np.reshape(bg_embs, (-1, 1, M, M)),2,axis=1)
        blobs['fg_norm'] = np.repeat(np.reshape(fg_norms, (-1, 1)),2,axis=1)
        blobs['bg_norm'] = np.repeat(np.reshape(bg_norms, (-1, 1)),2,axis=1)
        blobs['mask_emb_fg_labels'] = np.ones((fg_embs.shape[0],1),dtype=np.int32)
        blobs['mask_emb_bg_labels'] = np.zeros((bg_embs.shape[0], 1),dtype=np.int32)
        blobs['mask_emb_weights'] = np.vstack([fg_weights, bg_weights]).reshape((-1,1)).astype(np.float32)
    if cfg.BODY_UV_RCNN.USE_BOX_ALL_MASKS:
        blobs['body_masks_wrt_box'] = proposal_all_mask
        # blobs['semantic_segms'] = all_person_masks[np.newaxis]


def masks_to_embs(masks):
    # masks: batchsize, cls_num, M, M
    bs = masks.shape[0]
    fg_segs = []
    fg_weights = []
    bg_segs = []
    bg_weights = []

    for i in range(bs):
        # print('mask 0', masks[i,0].max(), masks[i,0].min())
        # print('mask 1', masks[i, 1].max(), masks[i, 1].min())
        fg_mask = (masks[i] > 0).astype(np.float32)
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
def expand_dp_targets(dp_body_part_target, target_sf, target_u, target_v, target_weight, target_uv_weight):
    idx_to_surface = {1: [1, 2],
                           2: [3],
                           3: [4],
                           4: [5],
                           5: [6],
                           6: [7, 9],
                           7: [8, 10],
                           8: [11, 13],
                           9: [12, 14],
                           10: [15, 17],
                           11: [16, 18],
                           12: [19, 21],
                           13: [20, 22],
                           14: [23, 24]}

    for part_id in range(1,15):

        surface_id = idx_to_surface[part_id]
        # if len(surface_id) > 1:
        #     continue

        part_map = (dp_body_part_target==part_id).astype(np.int32)
        if part_map.max()==0 or random.random() <= 0.5:
            continue
        surface_map = np.zeros_like(part_map)
        for s in surface_id:
            surface_map = np.logical_or(surface_map, target_sf==s)
        surface_map = surface_map.astype(np.int32)
        if surface_map.max()==0:
            continue
        diff_map = part_map - surface_map
        points_to_add = np.asarray(np.where(diff_map == 1)).transpose()
        labeled_points = np.asarray(np.where(surface_map == 1)).transpose()
        for i_point in range(points_to_add.shape[0]):
            point = points_to_add[i_point].reshape((1,2))
            dist = np.sqrt(np.sum((labeled_points - point)**2,axis=1))
            idx_target_point = np.argmin(dist,axis=0)
            loc_y, loc_x = labeled_points[idx_target_point,0],labeled_points[idx_target_point,1]
            point = point.reshape((2,))
            # expand I U V
            I = int(target_sf[loc_y, loc_x])
            target_sf[point[0], point[1]] = I
            U = target_u[I, loc_y, loc_x]
            V = target_v[I, loc_y, loc_x]
            u_to_add = (point[1]+1) * U / (loc_x+1)
            v_to_add = (point[0]+1) * V / (loc_y+1)
            target_u[I, point[0], point[1]] = u_to_add
            target_v[I, point[0], point[1]] = v_to_add
            # expand weight
            target_weight[point[0], point[1]] = 1
            target_uv_weight[I, point[0], point[1]] = 1
    return dp_body_part_target, target_sf, target_u, target_v, target_weight, target_uv_weight
