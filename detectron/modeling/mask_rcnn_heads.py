# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

"""Various network "heads" for predicting masks in Mask R-CNN.

The design is as follows:

... -> RoI ----\
                -> RoIFeatureXform -> mask head -> mask output -> loss
... -> Feature /
       Map

The mask head produces a feature representation of the RoI for the purpose
of mask prediction. The mask output module converts the feature representation
into real-valued (soft) masks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
from detectron.utils.net import get_group_gn
import detectron.modeling.ResNet as ResNet
import detectron.utils.blob as blob_utils


# ---------------------------------------------------------------------------- #
# Mask R-CNN outputs and losses
# ---------------------------------------------------------------------------- #
def add_mask_emb_outputs(model, blob_in, dim):
    if model.train:
        model.StopGradient('fg_mask', 'fg_mask')
        model.StopGradient('bg_mask', 'bg_mask')
        model.StopGradient('fg_norm', 'fg_norm')
        model.StopGradient('bg_norm', 'bg_norm')
    fill = (
        cfg.MRCNN.CONV_INIT
        if cfg.MRCNN.CLS_SPECIFIC_MASK else 'GaussianFill'
    )
    '''
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
    '''
    blob_in = model.Conv(
        blob_in,
        'mask_fcn_logits',
        dim,
        2,
        kernel=1,
        pad=0,
        stride=1,
        weight_init=(fill, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    
    if not model.train:
        blob_out = model.net.Sigmoid(blob_in, 'mask_fcn_probs')
        # mask_emb = model.net.Transpose(blob_in, ['mask_emb_transpose'], axes=(0, 2, 3, 1))
        # mask_emb,_ = model.net.Reshape(mask_emb, ['mask_emb','mask_emb_old_shape'], shape=(-1, dim))
        # mask_emb_logits = model.FC(
        #     mask_emb,
        #     'mask_emb_logits',
        #     dim,
        #     2,
        #     weight_init=gauss_fill(0.001),
        #     bias_init=const_fill(0.0)
        # )
        return blob_out

    fg_emb = model.net.Mul([blob_in, 'fg_mask'], ['fg_emb'])
    fg_emb = model.AveragePool(fg_emb, 'fg_emb_pool', kernel=28)
    # fg_emb = model.net.ReduceBackSum([fg_emb],['fg_emb_sum_1'],num_reduce_dims=2)
    # fg_emb = model.net.ReduceBackSum(fg_emb, 'fg_emb_sum_2')
    fg_emb,_ = model.net.Reshape(['fg_emb_pool'], ['fg_emb_pool_reshaped', 'fg_emb_shape'], shape=(-1, 2))
    fg_emb_normed = model.net.Mul([fg_emb, 'fg_norm'],'fg_emb_normed')
    '''
    fg_emb_normed = model.FC(
        fg_emb_normed,
        'fg_emb_normed_fc',
        dim,
        dim,
        weight_init=gauss_fill(0.001),
        bias_init=const_fill(0.0)
    )
    fg_emb_normed = model.Relu(fg_emb_normed, fg_emb_normed)
    '''
    bg_emb = model.net.Mul([blob_in, 'bg_mask'], ['bg_emb'])
    bg_emb = model.AveragePool(bg_emb, 'bg_emb_pool', kernel=28)
    bg_emb,_ = model.net.Reshape(['bg_emb_pool'], ['bg_emb_pool_reshaped', 'bg_emb_shape'], shape=(-1, 2))
    # bg_emb = model.net.ReduceBackSum(bg_emb,'bg_emb_sum_1')
    # bg_emb = model.net.ReduceBackSum(bg_emb, 'bg_emb_sum_2')
    bg_emb_normed = model.net.Mul([bg_emb, 'bg_norm'],'bg_emb_normed')
    '''
    bg_emb_normed = model.FC(
        bg_emb_normed,
        'bg_emb_normed_fc',
        dim,
        dim,
        weight_init=gauss_fill(0.001),
        bias_init=const_fill(0.0)
    )
    bg_emb_normed = model.Relu(bg_emb_normed, bg_emb_normed)
    '''
    mask_emb, _ = model.net.Concat([fg_emb_normed, bg_emb_normed],['mask_emb_logits','mask_emb_old_shape'], axis=0)
    # mask_emb_logits = mask_emb
    # mask_emb_logits = model.FC(
    #     mask_emb,
    #     'mask_emb_logits',
    #     dim,
    #     2,
    #     weight_init=gauss_fill(0.001),
    #     bias_init=const_fill(0.0)
    # )
    blob_out = blob_in
    return blob_out



def add_mask_rcnn_outputs(model, blob_in, dim):
    if cfg.MRCNN.DP_CASCADE_MASK_ON:

        return add_dp_cascaded_mask_outputs(model, blob_in, dim)
    if cfg.MRCNN.BBOX_CASCADE_MASK_ON:
        if cfg.MRCNN.USE_CLS_EMBS:
           return add_mask_emb_outputs(model, blob_in, dim)
        return add_cascaded_mask_outputs(model, blob_in, dim)
    """Add Mask R-CNN specific outputs: either mask logits or probs."""
    num_cls = cfg.MODEL.NUM_CLASSES if cfg.MRCNN.CLS_SPECIFIC_MASK else 1
    if cfg.MRCNN.USE_CLS_EMBS:
        return add_mask_emb_outputs(model, blob_in, dim)

    if cfg.MRCNN.USE_FC_OUTPUT:
        # Predict masks with a fully connected layer (ignore 'fcn' in the blob
        # name)
        blob_out = model.FC(
            blob_in,
            'mask_fcn_logits',
            dim,
            num_cls * cfg.MRCNN.RESOLUTION**2,
            weight_init=gauss_fill(0.001),
            bias_init=const_fill(0.0)
        )
    else:
        # Predict mask using Conv

        # Use GaussianFill for class-agnostic mask prediction; fills based on
        # fan-in can be too large in this case and cause divergence
        fill = (
            cfg.MRCNN.CONV_INIT
            if cfg.MRCNN.CLS_SPECIFIC_MASK else 'GaussianFill'
        )
        blob_out = model.Conv(
            blob_in,
            'mask_fcn_logits',
            dim,
            num_cls,
            kernel=1,
            pad=0,
            stride=1,
            weight_init=(fill, {'std': 0.001}),
            bias_init=const_fill(0.0)
        )

        if cfg.MRCNN.UPSAMPLE_RATIO > 1:
            blob_out = model.BilinearInterpolation(
                'mask_fcn_logits', 'mask_fcn_logits_up', num_cls, num_cls,
                cfg.MRCNN.UPSAMPLE_RATIO
            )

    if not model.train:  # == if test
        blob_out = model.net.Sigmoid(blob_out, 'mask_fcn_probs')

    return blob_out


def add_mask_rcnn_losses(model, blob_mask):
    """Add Mask R-CNN specific losses."""
    if cfg.MRCNN.DP_CASCADE_MASK_ON:
        return add_cascade_dp_mask_losses(model, blob_mask)
    '''
    probs_mask, loss_mask = model.net.SpatialSoftmaxWithLoss( \
                          [blob_mask, 'mask_labels','mask_weights'],\
                          ['probs_mask','loss_mask'], \
                           scale=cfg.BODY_UV_RCNN.INDEX_WEIGHTS / cfg.NUM_GPUS)
    '''
    loss_mask = model.net.SigmoidCrossEntropyLoss(
        [blob_mask, 'masks_int32'],
        'loss_mask',
        scale=model.GetLossScale() * cfg.MRCNN.WEIGHT_LOSS_MASK
    )
    
    losses = ['loss_mask']
    losses_to_gradients = [loss_mask]
    if cfg.MRCNN.BBOX_CASCADE_MASK_ON:
        loss_inter_mask = model.net.SigmoidCrossEntropyLoss(
            ['inter_person_mask', 'inter_masks_int32'],
            'loss_inter_mask',
            scale=model.GetLossScale() * cfg.MRCNN.WEIGHT_LOSS_MASK
        )
        losses += ['loss_inter_mask']
        losses_to_gradients += [loss_inter_mask]
    if cfg.MRCNN.USE_CLS_EMBS:
        #loss_mask_emb = model.net.SigmoidCrossEntropyLoss(
        #    ['mask_emb_logits', 'mask_emb_labels'],
        #    'loss_mask_emb',
        #    scale=model.GetLossScale() * cfg.MRCNN.WEIGHT_LOSS_MASK
        #)
        model.net.Concat(['mask_emb_fg_labels','mask_emb_bg_labels'],['mask_emb_labels','mask_emb_label_shape'],axis=0)
        mask_emb_prob, loss_mask_emb = model.net.SoftmaxWithLoss(
            ['mask_emb_logits', 'mask_emb_labels'], ['mask_emb_prob', 'loss_mask_emb'],
            scale=model.GetLossScale()
        )
        losses += ['loss_mask_emb']
        losses_to_gradients += [loss_mask_emb]
    loss_gradients = blob_utils.get_loss_gradients(model, losses_to_gradients)
    model.AddLosses(losses)
    return loss_gradients


def add_cascade_dp_mask_losses(model, blob_mask):
    model.net.Reshape(['body_mask_labels'], \
                      ['body_mask_labels_reshaped', 'body_uv_mask_labels_old_shape'], \
                      shape=(-1, cfg.BODY_UV_RCNN.HEATMAP_SIZE, cfg.BODY_UV_RCNN.HEATMAP_SIZE))

    model.net.Reshape(['body_uv_ann_weights'], \
                      ['body_uv_mask_weights_reshaped', 'body_uv_mask_weights_old_shape'], \
                      shape=(-1, cfg.BODY_UV_RCNN.HEATMAP_SIZE, cfg.BODY_UV_RCNN.HEATMAP_SIZE))
    # intermediate loss
    probs_inter_mask, loss_inter_mask = model.net.SpatialSoftmaxWithLoss( \
        ['inter_person_mask', 'body_masks_wrt_box', 'body_uv_mask_weights_reshaped'], \
        ['probs_inter_mask', 'loss_inter_mask'], \
        scale=cfg.BODY_UV_RCNN.INDEX_WEIGHTS / cfg.NUM_GPUS)
    # final mask loss
    probs_mask, loss_mask = model.net.SpatialSoftmaxWithLoss( \
        ['person_mask', 'body_mask_labels_reshaped', 'body_uv_mask_weights_reshaped'], \
        ['probs_mask', 'loss_mask'], \
        scale=cfg.BODY_UV_RCNN.INDEX_WEIGHTS / cfg.NUM_GPUS)
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_inter_mask, loss_mask])
    model.AddLosses(['loss_inter_mask', 'loss_mask'])
    return loss_gradients

# ---------------------------------------------------------------------------- #
# Mask heads
# ---------------------------------------------------------------------------- #

def mask_rcnn_fcn_head_v1up4convs(model, blob_in, dim_in, spatial_scale):
    """v1up design: 4 * (conv 3x3), convT 2x2."""
    return mask_rcnn_fcn_head_v1upXconvs(
        model, blob_in, dim_in, spatial_scale, 4
    )


def mask_rcnn_fcn_head_v1up4convs_gn(model, blob_in, dim_in, spatial_scale):
    """v1up design: 4 * (conv 3x3), convT 2x2, with GroupNorm"""
    return mask_rcnn_fcn_head_v1upXconvs_gn(
        model, blob_in, dim_in, spatial_scale, 4
    )


def mask_rcnn_fcn_head_v1up(model, blob_in, dim_in, spatial_scale):
    """v1up design: 2 * (conv 3x3), convT 2x2."""
    return mask_rcnn_fcn_head_v1upXconvs(
        model, blob_in, dim_in, spatial_scale, 2
    )


def mask_rcnn_fcn_head_v1upXconvs(
    model, blob_in, dim_in, spatial_scale, num_convs
):
    if cfg.MRCNN.DP_CASCADE_MASK_ON:
        print('DP CASCADED MASK')
        return add_roi_dp_cascade_uv_head(model, blob_in, dim_in, spatial_scale)
    if cfg.MRCNN.BBOX_CASCADE_MASK_ON:
        print('BBOX CASCADE MASK')
        return add_roi_mask_cascade_head(model, blob_in, dim_in, spatial_scale)
    """v1upXconvs design: X * (conv 3x3), convT 2x2."""
    print('General mask head')
    current = model.RoIFeatureTransform(
        blob_in,
        blob_out='_[mask]_roi_feat',
        blob_rois='mask_rois',
        method=cfg.MRCNN.ROI_XFORM_METHOD,
        resolution=cfg.MRCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.MRCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    dilation = cfg.MRCNN.DILATION
    dim_inner = cfg.MRCNN.DIM_REDUCED

    for i in range(num_convs):
        if cfg.MRCNN.USE_KPS_CONV:
           current = model.Conv(
                current,
                '_[mask]_fcn' + str(i + 1),
                dim_in,
                dim_inner,
                kernel=3,
                dilation=dilation,
                pad=1 * dilation,
                stride=1,
                weight='_[pose]_conv_fcn' + str(i + 1) + '_w',
                bias='_[pose]_conv_fcn' + str(i + 1) + '_b'
            )
        else:
            current = model.Conv(
                current,
                '_[mask]_fcn' + str(i + 1),
                dim_in,
                dim_inner,
                kernel=3,
                dilation=dilation,
                pad=1 * dilation,
                stride=1,
                weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
                bias_init=('ConstantFill', {'value': 0.})
            )
        current = model.Relu(current, current)
        dim_in = dim_inner

    # upsample layer
    model.ConvTranspose(
        current,
        'conv5_mask',
        dim_inner,
        dim_inner,
        kernel=2,
        pad=0,
        stride=2,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    blob_mask = model.Relu('conv5_mask', 'conv5_mask')

    return blob_mask, dim_inner


def mask_rcnn_fcn_head_v1upXconvs_gn(
    model, blob_in, dim_in, spatial_scale, num_convs
):
    """v1upXconvs design: X * (conv 3x3), convT 2x2, with GroupNorm"""
    current = model.RoIFeatureTransform(
        blob_in,
        blob_out='_mask_roi_feat',
        blob_rois='mask_rois',
        method=cfg.MRCNN.ROI_XFORM_METHOD,
        resolution=cfg.MRCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.MRCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    dilation = cfg.MRCNN.DILATION
    dim_inner = cfg.MRCNN.DIM_REDUCED

    for i in range(num_convs):
        current = model.ConvGN(
            current,
            '_mask_fcn' + str(i + 1),
            dim_in,
            dim_inner,
            group_gn=get_group_gn(dim_inner),
            kernel=3,
            pad=1 * dilation,
            stride=1,
            weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        current = model.Relu(current, current)
        dim_in = dim_inner

    # upsample layer
    model.ConvTranspose(
        current,
        'conv5_mask',
        dim_inner,
        dim_inner,
        kernel=2,
        pad=0,
        stride=2,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    blob_mask = model.Relu('conv5_mask', 'conv5_mask')

    return blob_mask, dim_inner


def mask_rcnn_fcn_head_v0upshare(model, blob_in, dim_in, spatial_scale):
    """Use a ResNet "conv5" / "stage5" head for mask prediction. Weights and
    computation are shared with the conv5 box head. Computation can only be
    shared during training, since inference is cascaded.

    v0upshare design: conv5, convT 2x2.
    """
    # Since box and mask head are shared, these must match
    assert cfg.MRCNN.ROI_XFORM_RESOLUTION == cfg.FAST_RCNN.ROI_XFORM_RESOLUTION

    if model.train:  # share computation with bbox head at training time
        dim_conv5 = 2048
        blob_conv5 = model.net.SampleAs(
            ['res5_2_sum', 'roi_has_mask_int32'],
            ['_[mask]_res5_2_sum_sliced']
        )
    else:  # re-compute at test time
        blob_conv5, dim_conv5 = add_ResNet_roi_conv5_head_for_masks(
            model,
            blob_in,
            dim_in,
            spatial_scale
        )

    dim_reduced = cfg.MRCNN.DIM_REDUCED

    blob_mask = model.ConvTranspose(
        blob_conv5,
        'conv5_mask',
        dim_conv5,
        dim_reduced,
        kernel=2,
        pad=0,
        stride=2,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),  # std only for gauss
        bias_init=const_fill(0.0)
    )
    model.Relu('conv5_mask', 'conv5_mask')

    return blob_mask, dim_reduced


def mask_rcnn_fcn_head_v0up(model, blob_in, dim_in, spatial_scale):
    """v0up design: conv5, deconv 2x2 (no weight sharing with the box head)."""
    blob_conv5, dim_conv5 = add_ResNet_roi_conv5_head_for_masks(
        model,
        blob_in,
        dim_in,
        spatial_scale
    )

    dim_reduced = cfg.MRCNN.DIM_REDUCED

    model.ConvTranspose(
        blob_conv5,
        'conv5_mask',
        dim_conv5,
        dim_reduced,
        kernel=2,
        pad=0,
        stride=2,
        weight_init=('GaussianFill', {'std': 0.001}),
        bias_init=const_fill(0.0)
    )
    blob_mask = model.Relu('conv5_mask', 'conv5_mask')

    return blob_mask, dim_reduced


def add_ResNet_roi_conv5_head_for_masks(model, blob_in, dim_in, spatial_scale):
    """Add a ResNet "conv5" / "stage5" head for predicting masks."""
    model.RoIFeatureTransform(
        blob_in,
        blob_out='_[mask]_pool5',
        blob_rois='mask_rois',
        method=cfg.MRCNN.ROI_XFORM_METHOD,
        resolution=cfg.MRCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.MRCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    dilation = cfg.MRCNN.DILATION
    stride_init = int(cfg.MRCNN.ROI_XFORM_RESOLUTION / 7)  # by default: 2

    s, dim_in = ResNet.add_stage(
        model,
        '_[mask]_res5',
        '_[mask]_pool5',
        3,
        dim_in,
        2048,
        512,
        dilation,
        stride_init=stride_init
    )

    return s, 2048


def add_roi_dp_cascade_uv_head(model, blob_in, dim_in, spatial_scale):
    """v1convX design: X * (conv)."""
    hidden_dim = cfg.BODY_UV_RCNN.CONV_HEAD_DIM
    kernel_size = cfg.BODY_UV_RCNN.CONV_HEAD_KERNEL
    pad_size = kernel_size // 2
    current = model.RoIFeatureTransform(
        blob_in,
        '_[mask]_roi_feat',
        blob_rois='body_uv_rois',
        method=cfg.BODY_UV_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.BODY_UV_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.BODY_UV_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    for i in range(cfg.BODY_UV_RCNN.NUM_STACKED_CONVS // 2):
        current = model.Conv(
            current,
            '_[mask]_conv_fcn' + str(i + 1),
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
            '_[mask]_conv_fcn' + str(i + 5),
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

def add_dp_cascaded_mask_outputs(model, blob_in, dim):
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
    # if not model.train:  # == if test
    #     blob_out = model.net.Sigmoid(blob_out, 'mask_fcn_probs')

    return blob_out


def add_roi_mask_cascade_head(model, blob_in, dim_in, spatial_scale):
    """v1convX design: X * (conv)."""
    hidden_dim = cfg.BODY_UV_RCNN.CONV_HEAD_DIM
    kernel_size = cfg.BODY_UV_RCNN.CONV_HEAD_KERNEL
    pad_size = kernel_size // 2
    current = model.RoIFeatureTransform(
        blob_in,
        '_[mask]_roi_feat',
        blob_rois='mask_rois',
        method=cfg.BODY_UV_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.BODY_UV_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.BODY_UV_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    for i in range(cfg.BODY_UV_RCNN.NUM_STACKED_CONVS // 2):
        current = model.Conv(
            current,
            '_[mask]_conv_fcn' + str(i + 1),
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
    # inter = model.ConvTranspose(inter, 'Inter_head_out_upres', hidden_dim, hidden_dim, cfg.BODY_UV_RCNN.DECONV_KERNEL,
    #                             pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1), stride=2,
    #                             weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
    #                             bias_init=('ConstantFill', {'value': 0.}))
    # inter = model.Relu(inter, inter)
    model.Conv(
        inter,
        'inter_person_mask',
        hidden_dim,
        1,
        1,
        stride=1,
        pad=0,
        weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.})
    )
    # add final head out
    for i in range(cfg.BODY_UV_RCNN.NUM_STACKED_CONVS // 2):
        current = model.Conv(
            current,
            '_[mask]_conv_fcn' + str(i + 5),
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
    # current = model.ConvTranspose(current, 'mask_head_out_upres', hidden_dim, hidden_dim,
    #                               cfg.BODY_UV_RCNN.DECONV_KERNEL, pad=int(cfg.BODY_UV_RCNN.DECONV_KERNEL / 2 - 1),
    #                               stride=2, weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
    #                               bias_init=('ConstantFill', {'value': 0.}))
    # current = model.Relu(current, current)
    current = model.Concat([current, inter], 'mask_head_out_concated', axis=1)

    current = model.Conv(
        current,
        'mask_head_out',
        hidden_dim*2,
        hidden_dim,
        1,
        stride=1,
        pad=0,
        weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.})
    )
    current = model.Relu(current, current)
    return current, hidden_dim

def add_cascaded_mask_outputs(model, blob_in, dim):
    blob_out = model.Conv(
            blob_in,
            'mask_fcn_logits',
            dim,
            2,
            1,
            stride=1,
            pad=0,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.})
    )
    if not model.train:  # == if test
        blob_out = model.net.Sigmoid(blob_out, 'mask_fcn_probs')

    return blob_out