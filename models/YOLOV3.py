import torch
import torch.nn as nn
from .ReferenceNet import ReferenceNet


class YOLOV3(nn.Module):
    def __init__(self, B=3, C=20):
        super(YOLOV3, self).__init__()
        in_channels = 3
        out_channels = 1024
        add_channels = 256
        yolo_channels = (5 + C) * B
        self.features = ReferenceNet(in_channels=in_channels, out_channels=out_channels)
        self.additional = nn.Sequential(
            nn.Conv2d(out_channels, add_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(add_channels),
            nn.LeakyReLU(0.1)
        )

        # yolo2
        self.yolo_layer2_neck = nn.Sequential(
            nn.Conv2d(add_channels, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1)
        )
        self.yolo_layer2_head = nn.Sequential(
            nn.Conv2d(512, yolo_channels, kernel_size=1, stride=1, padding=0)
        )

        # yolo1
        self.yolo_layer1_neck1 = nn.Sequential(
            nn.Conv2d(add_channels, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.yolo_layer1_neck2 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1)
        )
        self.yolo_layer1_head = nn.Sequential(
            nn.Conv2d(256, yolo_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x, route = self.features(x)
        x = self.additional(x)

        # yolo2
        yolo_layer2_neck = self.yolo_layer2_neck(x)
        yolo_layer2_head = self.yolo_layer2_head(yolo_layer2_neck)

        # yolo1
        yolo_layer1_neck = self.yolo_layer1_neck2(torch.cat([self.yolo_layer1_neck1(x), route], dim=1))
        yolo_layer1_head = self.yolo_layer1_head(yolo_layer1_neck)

        return [yolo_layer1_head, yolo_layer2_head]