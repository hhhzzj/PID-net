# PID-net: Progressive Instance Distillation Network for Dense Human Pose Estimation In-the-Wild
-------
Code coming soon.
![fig](https://github.com/hhhzzj/PID-net/blob/master/result_5.gif)
![fig](https://github.com/hhhzzj/PID-net/blob/master/result_7.gif)
Dense human pose estimation aims at mapping all human
pixels of an RGB image to a 3D surface of the human
body. Despite of recent progress, a severe problem faced by
existing methods is that the generated instance-level human
body is largely inaccurate in the presence of partially occlusions,
overlapping and scales of variations, resulting in
an imprecise densepose estimation. In this paper, we propose
a novel framework, named Progressive Instance Distillation
network (PID-net), for human instance-level analysis
in-the-wild. It enables the mapping of a 2D image to
a human body 3D surface (image-to-surface) based representations
(i.e., Index-to-Patch (I), and UV coordinates),
by elegantly handling instance-level human body distillation.
Specifically, we design our PID-net as follows: 1)
We equip IUV-RCNN branch with a semantic-to-instance
distillation modular by involving intermediate supervision
to effectively segment multi-instances in one bounding box,
which can help distinguishing hard foreground from complex
background; and 2) we adopt an instance-aware learning
objective, which can further distinguish the target instance
from other interference instances. Extensive experiments
on the large-scale and challenging dataset (i.e.,
DensePose-COCO) demonstrate the effectiveness of our
proposed method. Our method surpasses the state-of-theart
by a large margin.

![fig](https://github.com/hhhzzj/PID-net/blob/master/result.png)

Figure 1. Our PID-net prediction results. It aims to estimate dense correspondences from a 2D image in the wild to a 3D surface-based
presentations (i.e., Index-to-Patch, and specific U and V coordinates) of a human body. Each example is arranged from left to right with the
following order: an input image and its corresponding 3D Index-to Patch (the patch ID of a pixel), U and V coordinates (the coordinates of
a pixel in its corresponding patch).

Training a model
-------
This example shows how to trian a model on the DensePose-COCO dataset. We can use different structure to train using different config. The model uses a ResNet-50-FPN backbone with an end-to-end trianing approach.

```
python2 tools/train_net.py \
    --cfg configs/coco_exp_configs/DensePose_ResNet50_FPN_cascade_mask_dp_s1x-e2e.yaml \
    OUTPUT_DIR /tmp/detectron-output
```

Testing a pretrianed model
-------
Before testing, you should make sure that you have downloaded the pretrianed model. This example shows how to run a pretrained model using a single GPU for inference. 
```
python2 tools/test_net.py \
    --cfg configs/coco_exp_configs/DensePose_ResNet50_FPN_cascade_mask_dp_s1x-e2e.yaml \
    TEST.WEIGHTS /the/dir/of/your/trained/model \
    NUM_GPUS 1
```



