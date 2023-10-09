import numpy as np
import torch.nn as nn
from detect import model_init

def load_conv_model(module, f):
    conv_layer = module[0]
    if len(module) > 1 and isinstance(module[1], nn.BatchNorm2d):
        bn_layer = module[1]
        # bn bias
        num_b = bn_layer.bias.numel()
        bn_b = bn_layer.bias.data.view(num_b).numpy()
        bn_b.tofile(f)
        # bn weights
        num_w = bn_layer.weight.numel()
        bn_w = bn_layer.weight.data.view(num_w).numpy()
        bn_w.tofile(f)
        # bn running mean
        num_rm = bn_layer.running_mean.numel()
        bn_rm = bn_layer.running_mean.data.view(num_rm).numpy()
        bn_rm.tofile(f)
        # bn running var
        num_rv = bn_layer.running_var.numel()
        bn_rv = bn_layer.running_var.data.view(num_rv).numpy()
        bn_rv.tofile(f)
    else:
        # conv bias
        num_b = conv_layer.bias.numel()
        conv_b = conv_layer.bias.data.view(num_b).numpy()
        conv_b.tofile(f)
    # conv weights
    num_w = conv_layer.weight.numel()
    conv_w = conv_layer.weight.data.view(num_w).numpy()
    conv_w.tofile(f)

print('load pytorch model ... ')
checkpoint_path = 'weights/yolov3_final.pth'
B, C = 3, 20
model = model_init(checkpoint_path, B, C)

print('convert to darknet ... ')
with open('weights/yolov3-tiny-final.weights', 'wb') as f:
    np.asarray([0, 2, 0, 32013312, 0], dtype=np.int32).tofile(f)
    for module in model.features.features:
        if isinstance(module[0], nn.Conv2d):
            load_conv_model(module, f)

    # addition module
    load_conv_model(model.additional, f)
    # yolo_layer1_neck
    load_conv_model(model.yolo_layer2_neck, f)
    # yolo_layer1_head
    load_conv_model(model.yolo_layer2_head, f)
    # yolo_layer2_neck1
    load_conv_model(model.yolo_layer1_neck1, f)
    # yolo_layer2_neck2
    load_conv_model(model.yolo_layer1_neck2, f)
    # yolo_layer2_head
    load_conv_model(model.yolo_layer1_head, f)

print('done!')