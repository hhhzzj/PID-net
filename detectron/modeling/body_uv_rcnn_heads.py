# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core

from detectron.core.config import cfg

import detectron.modeling.ResNet as ResNet
import detectron.utils.blob as blob_utils
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
# ---------------------------------------------------------------------------- #
# Body UV heads
# ---------------------------------------------------------------------------- #
def add_mask_emb_outputs(model, blob_in, dim):
    if model.train:
        model.StopGradient('fg_mask', 'fg_mask')
        model.StopGradient('bg_mask', 'bg_mask')
        model.StopGradient('fg_norm', 'fg_norm')
        model.StopGradient('bg_norm', 'bg_norm')
    '''
    blob_in = model.ConvTranspose(blob_in, 'mask_emb_lowres', dim, dim,cfg.BODY_UV_RCNN.DECONV_KERNEL, pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1), stride=2, weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}), bias_init=('ConstantFill', {'value': 0.}))
    blob_in = model.Relu(blob_in,blob_in)
    blob_in = model.BilinearInterpolation(
        blob_in, 'mask_emb_up', dim, dim, 2
    )
    blob_in = model.Conv(
        blob_in,
        'mask_fcn_emb',
        dim,
        dim,
        kernel=3,
        pad=1,
        stride=1,
        weight_init=((cfg.MRCNN.CONV_INIT), {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    blob_in = model.Relu(blob_in,blob_in)
    pmask = model.Conv(
        blob_in,
        'person_mask',
        dim,
        2,
        kernel=3,
        pad=1,
        stride=1,
        weight_init=((cfg.MRCNN.CONV_INIT), {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    blob_in = pmask
    '''
    if not model.train:
        return blob_in,None
    fg_emb = model.net.Mul([blob_in, 'fg_mask'], ['fg_emb'])
    fg_emb = model.AveragePool(fg_emb, 'fg_emb_pool', kernel=56)
    # fg_emb = model.net.ReduceBackSum([fg_emb],['fg_emb_sum_1'],num_reduce_dims=2)
    # fg_emb = model.net.ReduceBackSum(fg_emb, 'fg_emb_sum_2')
    fg_emb,_ = model.net.Reshape(['fg_emb_pool'], ['fg_emb_pool_reshaped', 'fg_emb_shape'], shape=(-1, 2))
    fg_emb_normed = model.net.Mul([fg_emb, 'fg_norm'],'fg_emb_normed')
#    fg_emb_normed = model.FC(
#        fg_emb_normed,
#        'fg_emb_normed_fc',
#        2,
#        2,
#        weight_init=gauss_fill(0.001),
#        bias_init=const_fill(0.0)
#    )
#    fg_emb_normed = model.Relu(fg_emb_normed, fg_emb_normed)

    bg_emb = model.net.Mul([blob_in, 'bg_mask'], ['bg_emb'])
    bg_emb = model.AveragePool(bg_emb, 'bg_emb_pool', kernel=56)
    bg_emb,_ = model.net.Reshape(['bg_emb_pool'], ['bg_emb_pool_reshaped', 'bg_emb_shape'], shape=(-1,2))
    # bg_emb = model.net.ReduceBackSum(bg_emb,'bg_emb_sum_1')
    # bg_emb = model.net.ReduceBackSum(bg_emb, 'bg_emb_sum_2')
    bg_emb_normed = model.net.Mul([bg_emb, 'bg_norm'],'bg_emb_normed')
#    bg_emb_normed = model.FC(
#        bg_emb_normed,
#        'bg_emb_normed_fc',
#        2,
#        2,
#        weight_init=gauss_fill(0.001),
#        bias_init=const_fill(0.0)
#    )
#    bg_emb_normed = model.Relu(bg_emb_normed, bg_emb_normed)

    mask_emb_logits, _ = model.net.Concat([fg_emb_normed, bg_emb_normed],['mask_emb_logits','mask_emb_old_shape'], axis=0)
    # mask_emb_logits = mask_emb
    '''
    mask_emb_logits = model.FC(
        mask_emb,
        'mask_emb_logits',
        dim,
        2,
        weight_init=gauss_fill(0.001),
        bias_init=const_fill(0.0)
    )
    '''
   # model.StopGradient('mask_emb_logits', 'mask_emb_logits')
    return mask_emb_logits

def add_mask_match_heads(model):
    # construct inputs
    model.net.Concat(['instances_data','person_mask'],['matched_fake_masks','matched_fake_mask_shape'],axis=1)
    model.net.Concat(['instances_fake_data', 'person_mask'],['unmatched_fake_masks','unmatched_fake_mask_shape'], axis=1)
    current, _ = model.net.Concat(['matched_real_masks','matched_fake_masks','unmatched_fake_masks'],['dnet_inputs','dnet_inputs_shape'],axis=0)
    dim = 5
    hidden_dim = 64 
    for i in range(4):
        current = model.Conv(
            current,
            'dnet_conv_fcn' + str(i + 1),
            dim,
            hidden_dim,
            3,
            stride=1,
            pad=1,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        current = model.Relu(current, current)
        if i < 3:
           current = model.MaxPool(current, 'dnet_pool'+str(i+1), kernel=2,stride=2)
        else:
           current = model.AveragePool(current, 'dnet_pool'+str(i+1), kernel=7) 
        dim = hidden_dim
        hidden_dim *= 2
    dlogits = model.FC(
        current,
        'dnet_logits',
        hidden_dim,
        2,
        weight_init=gauss_fill(0.001),
        bias_init=const_fill(0.0)
    )
    model.net.Concat(['dnet_matched_labels', 'dnet_unmatched_labels','dnet_unmatched_labels'],['dnet_labels','dnet_labels_shape'],axis=0)
    dnet_prob, loss_dnet = model.net.SoftmaxWithLoss(
             ['dnet_logits', 'dnet_labels'], ['dnet_prob', 'loss_dnet'],
             scale=cfg.BODY_UV_RCNN.INDEX_WEIGHTS / cfg.NUM_GPUS)
    return dlogits, loss_dnet

 

def add_body_uv_outputs(model, blob_in, dim, pref=''):
    if cfg.BODY_UV_RCNN.DP_CASCADE_MASK_ON:
       blob_U,blob_V,blob_Index,blob_Ann_Index = add_dp_cascaded_mask_outputs(model, blob_in, dim)
       if cfg.BODY_UV_RCNN.USE_CLS_EMBS:
          mask_emb_logits = add_mask_emb_outputs(model, 'person_mask', 2)
       return blob_U,blob_V,blob_Index,blob_Ann_Index
    if cfg.BODY_UV_RCNN.USE_AMA_NET:
        return add_ama_body_uv_outputs(model, blob_in, dim)
    if cfg.BODY_UV_RCNN.USE_CLS_EMBS:
        person_mask, mask_emb_logits = add_mask_emb_outputs(model, blob_in, dim)
    
    ####
    model.ConvTranspose(blob_in, 'AnnIndex_lowres'+pref, dim, 15,cfg.BODY_UV_RCNN.DECONV_KERNEL, pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1), stride=2, weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}), bias_init=('ConstantFill', {'value': 0.}))    
    ####
    model.ConvTranspose(blob_in, 'Index_UV_lowres'+pref, dim, cfg.BODY_UV_RCNN.NUM_PATCHES+1,cfg.BODY_UV_RCNN.DECONV_KERNEL, pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1), stride=2, weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}), bias_init=('ConstantFill', {'value': 0.}))    
    ####
    model.ConvTranspose(
        blob_in, 'U_lowres'+pref, dim, (cfg.BODY_UV_RCNN.NUM_PATCHES+1),
        cfg.BODY_UV_RCNN.DECONV_KERNEL,
        pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1),
        stride=2,
        weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
        bias_init=('ConstantFill', {'value': 0.}))
    #####
    model.ConvTranspose(
            blob_in, 'V_lowres'+pref, dim, cfg.BODY_UV_RCNN.NUM_PATCHES+1,
            cfg.BODY_UV_RCNN.DECONV_KERNEL,
            pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1),
            stride=2,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.}))
    ######
    model.ConvTranspose(
        blob_in, 'person_mask_lowres' + pref, dim, 2,
        cfg.BODY_UV_RCNN.DECONV_KERNEL,
        pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1),
        stride=2,
        weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
        bias_init=('ConstantFill', {'value': 0.}))
    ####
    blob_Ann_Index = model.BilinearInterpolation('AnnIndex_lowres'+pref, 'AnnIndex'+pref,  cfg.BODY_UV_RCNN.NUM_PATCHES+1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1, cfg.BODY_UV_RCNN.UP_SCALE)
    blob_Index = model.BilinearInterpolation('Index_UV_lowres'+pref, 'Index_UV'+pref,  cfg.BODY_UV_RCNN.NUM_PATCHES+1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1, cfg.BODY_UV_RCNN.UP_SCALE)
    blob_U = model.BilinearInterpolation('U_lowres'+pref, 'U_estimated'+pref,  cfg.BODY_UV_RCNN.NUM_PATCHES+1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1, cfg.BODY_UV_RCNN.UP_SCALE)
    blob_V = model.BilinearInterpolation('V_lowres'+pref, 'V_estimated'+pref,  cfg.BODY_UV_RCNN.NUM_PATCHES+1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1, cfg.BODY_UV_RCNN.UP_SCALE)
    blob_p = model.BilinearInterpolation('person_mask_lowres' + pref, 'person_mask', 2, 2, cfg.BODY_UV_RCNN.UP_SCALE)
    ###
    return blob_U,blob_V,blob_Index,blob_Ann_Index


def add_body_uv_losses(model, pref=''):
    if cfg.BODY_UV_RCNN.USE_AMA_NET:
        return add_body_uv_ama_losses(model, pref)
    ## Reshape for GT blobs.
    model.net.Reshape( ['body_uv_X_points'], ['X_points_reshaped'+pref, 'X_points_shape'+pref],  shape=( -1 ,1 ) )
    model.net.Reshape( ['body_uv_Y_points'], ['Y_points_reshaped'+pref, 'Y_points_shape'+pref],  shape=( -1 ,1 ) )
    model.net.Reshape( ['body_uv_I_points'], ['I_points_reshaped'+pref, 'I_points_shape'+pref],  shape=( -1 ,1 ) )
    model.net.Reshape( ['body_uv_Ind_points'], ['Ind_points_reshaped'+pref, 'Ind_points_shape'+pref],  shape=( -1 ,1 ) )
    ## Concat Ind,x,y to get Coordinates blob.
    model.net.Concat( ['Ind_points_reshaped'+pref,'X_points_reshaped'+pref, \
                       'Y_points_reshaped'+pref],['Coordinates'+pref,'Coordinate_Shapes'+pref ], axis = 1 )
    ##
    ### Now reshape UV blobs, such that they are 1x1x(196*NumSamples)xNUM_PATCHES 
    ## U blob to
    ##
    model.net.Reshape(['body_uv_U_points'], \
                      ['U_points_reshaped'+pref, 'U_points_old_shape'+pref],\
                      shape=(-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1,196))
    model.net.Transpose(['U_points_reshaped'+pref] ,['U_points_reshaped_transpose'+pref],axes=(0,2,1) )
    model.net.Reshape(['U_points_reshaped_transpose'+pref], \
                      ['U_points'+pref, 'U_points_old_shape2'+pref], \
                      shape=(1,1,-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1))
    ## V blob
    ##
    model.net.Reshape(['body_uv_V_points'], \
                      ['V_points_reshaped'+pref, 'V_points_old_shape'+pref],\
                      shape=(-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1,196))
    model.net.Transpose(['V_points_reshaped'+pref] ,['V_points_reshaped_transpose'+pref],axes=(0,2,1) )
    model.net.Reshape(['V_points_reshaped_transpose'+pref], \
                      ['V_points'+pref, 'V_points_old_shape2'+pref], \
                      shape=(1,1,-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1))
    ###
    ## UV weights blob
    ##
    model.net.Reshape(['body_uv_point_weights'], \
                      ['Uv_point_weights_reshaped'+pref, 'Uv_point_weights_old_shape'+pref],\
                      shape=(-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1,196))
    model.net.Transpose(['Uv_point_weights_reshaped'+pref] ,['Uv_point_weights_reshaped_transpose'+pref],axes=(0,2,1) )
    model.net.Reshape(['Uv_point_weights_reshaped_transpose'+pref], \
                      ['Uv_point_weights'+pref, 'Uv_point_weights_old_shape2'+pref], \
                      shape=(1,1,-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1))

    #####################
    ###  Pool IUV for points via bilinear interpolation.
    model.PoolPointsInterp(['U_estimated','Coordinates'+pref], ['interp_U'+pref])
    model.PoolPointsInterp(['V_estimated','Coordinates'+pref], ['interp_V'+pref])
    model.PoolPointsInterp(['Index_UV'+pref,'Coordinates'+pref], ['interp_Index_UV'+pref])

    ## Reshape interpolated UV coordinates to apply the loss.
    
    model.net.Reshape(['interp_U'+pref], \
                      ['interp_U_reshaped'+pref, 'interp_U_shape'+pref],\
                      shape=(1, 1, -1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1))
    
    model.net.Reshape(['interp_V'+pref], \
                      ['interp_V_reshaped'+pref, 'interp_V_shape'+pref],\
                      shape=(1, 1, -1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1))
    ###

    ### Do the actual labels here !!!!
    model.net.Reshape( ['body_uv_ann_labels'],    \
                      ['body_uv_ann_labels_reshaped'   +pref, 'body_uv_ann_labels_old_shape'+pref], \
                      shape=(-1, cfg.BODY_UV_RCNN.HEATMAP_SIZE , cfg.BODY_UV_RCNN.HEATMAP_SIZE))
    
    model.net.Reshape( ['body_uv_ann_weights'],   \
                      ['body_uv_ann_weights_reshaped'   +pref, 'body_uv_ann_weights_old_shape'+pref], \
                      shape=( -1 , cfg.BODY_UV_RCNN.HEATMAP_SIZE , cfg.BODY_UV_RCNN.HEATMAP_SIZE))
    ###
    model.net.Cast( ['I_points_reshaped'+pref], ['I_points_reshaped_int'+pref], to=core.DataType.INT32)
    model.net.Reshape(['body_mask_labels'], \
                      ['body_mask_labels_reshaped', 'body_uv_mask_labels_old_shape'], \
                      shape=(-1, cfg.BODY_UV_RCNN.HEATMAP_SIZE, cfg.BODY_UV_RCNN.HEATMAP_SIZE))
    # stop grad for ann
    # model.StopGradient('AnnIndex', 'AnnIndex')        
    ### Now add the actual losses 
    ## The mask segmentation loss (dense)
    probs_seg_AnnIndex, loss_seg_AnnIndex = model.net.SpatialSoftmaxWithLoss( \
                          ['AnnIndex'+pref, 'body_uv_ann_labels_reshaped'+pref,'body_uv_ann_weights_reshaped'+pref],\
                          ['probs_seg_AnnIndex'+pref,'loss_seg_AnnIndex'+pref], \
                           scale=cfg.BODY_UV_RCNN.INDEX_WEIGHTS / cfg.NUM_GPUS)
    ## Point Patch Index Loss.
    probs_IndexUVPoints, loss_IndexUVPoints = model.net.SoftmaxWithLoss(\
                          ['interp_Index_UV'+pref,'I_points_reshaped_int'+pref],\
                          ['probs_IndexUVPoints'+pref,'loss_IndexUVPoints'+pref], \
                          scale=cfg.BODY_UV_RCNN.PART_WEIGHTS / cfg.NUM_GPUS, spatial=0)

    probs_IndexUVPoints_ori, loss_IndexUVPoints_ori =model.net.SpatialSoftmaxWithLoss( \
                          ['Index_UV', 'body_uv_Index_targets','body_uv_Index_targets_weights'],\
                          ['probs_IndexUVPoints_ori'+pref,'loss_IndexUVPoints_ori'+pref], \
                           scale=0.1*cfg.BODY_UV_RCNN.INDEX_WEIGHTS / cfg.NUM_GPUS)
    ## U and V point losses.
    loss_Upoints = model.net.SmoothL1Loss( \
                          ['interp_U_reshaped'+pref, 'U_points'+pref, \
                               'Uv_point_weights'+pref, 'Uv_point_weights'+pref], \
                          'loss_Upoints'+pref, \
                            scale=cfg.BODY_UV_RCNN.POINT_REGRESSION_WEIGHTS  / cfg.NUM_GPUS)
    loss_Upoints_ori = model.net.SmoothL1Loss( \
        ['U_estimated', 'body_uv_U_targets', \
         'body_uv_weights', 'body_uv_weights'], \
        'loss_Upoints_ori' + pref, \
        scale=0.1*cfg.BODY_UV_RCNN.POINT_REGRESSION_WEIGHTS / cfg.NUM_GPUS)
    
    loss_Vpoints = model.net.SmoothL1Loss( \
                          ['interp_V_reshaped'+pref, 'V_points'+pref, \
                               'Uv_point_weights'+pref, 'Uv_point_weights'+pref], \
                          'loss_Vpoints'+pref, scale=cfg.BODY_UV_RCNN.POINT_REGRESSION_WEIGHTS / cfg.NUM_GPUS)
    loss_Vpoints_ori = model.net.SmoothL1Loss( \
        ['V_estimated', 'body_uv_V_targets', \
         'body_uv_weights', 'body_uv_weights' ], \
        'loss_Vpoints_ori' + pref, scale=0.1*cfg.BODY_UV_RCNN.POINT_REGRESSION_WEIGHTS / cfg.NUM_GPUS)

    probs_mask, loss_mask = model.net.SpatialSoftmaxWithLoss( \
        ['person_mask', 'body_mask_labels_reshaped', 'body_uv_ann_weights_reshaped'], \
        ['probs_mask', 'loss_mask'], \
        scale=cfg.BODY_UV_RCNN.INDEX_WEIGHTS / cfg.NUM_GPUS)
    ## Add the losses.
    losses = ['loss_Upoints' + pref, 'loss_Vpoints' + pref, 'loss_seg_AnnIndex' + pref, 'loss_IndexUVPoints' + pref,'loss_mask'
              ,'loss_Vpoints_ori','loss_Upoints_ori','loss_IndexUVPoints_ori']
    losses_to_gradients = [loss_Upoints, loss_Vpoints, loss_seg_AnnIndex, loss_IndexUVPoints, loss_mask
                           ,loss_Vpoints_ori,loss_Upoints_ori,loss_IndexUVPoints_ori]

    #### additional loss #######
    if cfg.BODY_UV_RCNN.USE_CLS_EMBS:
        
        model.net.Concat(['mask_emb_fg_labels', 'mask_emb_bg_labels'],['mask_emb_labels','mask_emb_labels_shape'],axis=0)
        mask_emb_prob, loss_mask_emb = model.net.SoftmaxWithLoss(
             ['mask_emb_logits', 'mask_emb_labels'], ['mask_emb_prob', 'loss_mask_emb'],
             scale=cfg.BODY_UV_RCNN.INDEX_WEIGHTS / cfg.NUM_GPUS)
    
    if cfg.BODY_UV_RCNN.DP_CASCADE_MASK_ON:
        # intermediate loss

        probs_inter_mask, loss_inter_mask = model.net.SpatialSoftmaxWithLoss( \
            ['inter_person_mask', 'body_masks_wrt_box', 'body_uv_ann_weights_reshaped'], \
            ['probs_inter_mask', 'loss_inter_mask'], \
            scale=cfg.BODY_UV_RCNN.INTER_WEIGHTS*cfg.BODY_UV_RCNN.INDEX_WEIGHTS / cfg.NUM_GPUS)
        
        losses_to_gradients += [loss_inter_mask]
        losses += ['loss_inter_mask']
    if cfg.BODY_UV_RCNN.USE_CLS_EMBS:
        
        losses_to_gradients += [loss_mask_emb]
        losses += ['loss_mask_emb']
    loss_gradients = blob_utils.get_loss_gradients(model, losses_to_gradients)
    model.losses = list(set(model.losses + losses))

    return loss_gradients


# ---------------------------------------------------------------------------- #
# Body UV heads
# ---------------------------------------------------------------------------- #

def add_ResNet_roi_conv5_head_for_bodyUV(
        model, blob_in, dim_in, spatial_scale
):
    """Add a ResNet "conv5" / "stage5" head for body UV prediction."""
    model.RoIFeatureTransform(
        blob_in, '_[body_uv]_pool5',
        blob_rois='body_uv_rois',
        method=cfg.BODY_UV_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.BODY_UV_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.BODY_UV_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)
    # Using the prefix '_[body_uv]_' to 'res5' enables initializing the head's
    # parameters using pretrained 'res5' parameters if given (see
    # utils.net.initialize_from_weights_file)
    s, dim_in = ResNet.add_stage(
        model,
        '_[body_uv]_res5',
        '_[body_uv]_pool5',
        3,
        dim_in,
        2048,
        512,
        cfg.BODY_UV_RCNN.DILATION,
        stride_init=int(cfg.BODY_UV_RCNN.ROI_XFORM_RESOLUTION / 7)
    )
    return s, 2048


def add_roi_body_uv_head_v1convX(model, blob_in, dim_in, spatial_scale):
    """v1convX design: X * (conv)."""
    if cfg.BODY_UV_RCNN.DP_CASCADE_MASK_ON:
        return add_roi_dp_cascade_uv_head(model, blob_in, dim_in, spatial_scale)
    if cfg.BODY_UV_RCNN.USE_AMA_NET:
        return add_roi_body_uv_ama_head(model, blob_in, dim_in, spatial_scale)
    hidden_dim = cfg.BODY_UV_RCNN.CONV_HEAD_DIM
    kernel_size = cfg.BODY_UV_RCNN.CONV_HEAD_KERNEL
    pad_size = kernel_size // 2
    current = model.RoIFeatureTransform(
        blob_in,
        '_[body_uv]_roi_feat',
        blob_rois='body_uv_rois',
        method=cfg.BODY_UV_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.BODY_UV_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.BODY_UV_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    for i in range(cfg.BODY_UV_RCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current,
            'body_conv_fcn' + str(i + 1),
            dim_in,
            hidden_dim,
            kernel_size,
            stride=1,
            pad=pad_size,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        current = model.Relu(current, current)
        dim_in = hidden_dim

    return current, hidden_dim


def add_roi_dp_cascade_uv_head(model, blob_in, dim_in, spatial_scale):
    """v1convX design: X * (conv)."""
    hidden_dim = cfg.BODY_UV_RCNN.CONV_HEAD_DIM
    kernel_size = cfg.BODY_UV_RCNN.CONV_HEAD_KERNEL
    pad_size = kernel_size // 2

    k_max = cfg.FPN.ROI_MAX_LEVEL  # coarsest level of pyramid
    k_min = cfg.FPN.ROI_MIN_LEVEL  # finest level of pyramid
    bl_out_list = []

    current = model.RoIFeatureTransform(
        blob_in,
        '_[body_uv]_roi_feat',
        blob_rois='body_uv_rois',
        method=cfg.BODY_UV_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.BODY_UV_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.BODY_UV_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    for i in range(cfg.BODY_UV_RCNN.NUM_STACKED_CONVS // 2):
        current = model.Conv(
            current,
            'body_conv_fcn' + str(i + 1),
            dim_in,
            hidden_dim,
            kernel_size,
            stride=1,
            pad=pad_size,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        current = model.Relu(current, current)
        dim_in = hidden_dim
    # add intermediate out
    inter = model.ConvTranspose(current, 'Inter_head_out_lowres', hidden_dim, hidden_dim,
                                cfg.BODY_UV_RCNN.DECONV_KERNEL, pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1),
                                stride=2, weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
                                bias_init=('ConstantFill', {'value': 0.}))
    inter = model.Relu(inter, inter)
    inter = model.ConvTranspose(inter, 'Inter_head_out_upres', hidden_dim, hidden_dim, cfg.BODY_UV_RCNN.DECONV_KERNEL,
                                pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1), stride=2,
                                weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
                                bias_init=('ConstantFill', {'value': 0.}))
    inter = model.Relu(inter, inter)
    model.Conv(
        inter,
        'inter_person_mask',
        hidden_dim,
        2,
        3,
        stride=1,
        pad=1,
        weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.})
    )
    # add final head out
    for i in range(cfg.BODY_UV_RCNN.NUM_STACKED_CONVS // 2):
        current = model.Conv(
            current,
            'body_conv_fcn' + str(i + 5),
            dim_in,
            hidden_dim,
            kernel_size,
            stride=1,
            pad=pad_size,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        current = model.Relu(current, current)
        dim_in = hidden_dim
    current = model.ConvTranspose(current, 'mask_head_out_lowres', hidden_dim, hidden_dim,
                                  cfg.BODY_UV_RCNN.DECONV_KERNEL, pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1),
                                  stride=2, weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
                                  bias_init=('ConstantFill', {'value': 0.}))
    current = model.Relu(current, current)
    current = model.ConvTranspose(current, 'mask_head_out_upres', hidden_dim, hidden_dim,
                                  cfg.BODY_UV_RCNN.DECONV_KERNEL, pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1),
                                  stride=2, weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
                                  bias_init=('ConstantFill', {'value': 0.}))
    current = model.Relu(current, current)
    current = model.Concat([current, inter], 'mask_head_out_concated', axis=1)

    current = model.Conv(
        current,
        'mask_head_out',
        hidden_dim * 2,
        hidden_dim,
        1,
        stride=1,
        pad=0,
        weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.})
    )
    current = model.Relu(current, current)
    return current, hidden_dim

def add_dp_cascaded_mask_outputs(model, blob_in, dim, pref=''):
    blob_out = model.Conv(
            blob_in,
            'person_mask',
            dim,
            2,
            3,
            stride=1,
            pad=1,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.})
    )
    blob_Index  = model.Conv(
            blob_in,
            'Index_UV'+pref,
            dim,
            cfg.BODY_UV_RCNN.NUM_PATCHES + 1,
            3,
            stride=1,
            pad=1,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.})
    )
    blob_U = model.Conv(
            blob_in,
            'U_estimated'+pref,
            dim,
            cfg.BODY_UV_RCNN.NUM_PATCHES + 1,
            3,
            stride=1,
            pad=1,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.})
    )
    blob_V  = model.Conv(
            blob_in,
            'V_estimated'+pref,
            dim,
            cfg.BODY_UV_RCNN.NUM_PATCHES + 1,
            3,
            stride=1,
            pad=1,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.})
    )  
    blob_Ann_Index  = model.Conv(
        blob_in,
        'AnnIndex'+pref,
        dim,
        cfg.BODY_UV_RCNN.NUM_BODY_PARTS + 1,
        3,
        stride=1,
        pad=1,
        weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
        bias_init=('ConstantFill', {'value': 0.})
    )
    return blob_U,blob_V,blob_Index,blob_Ann_Index
