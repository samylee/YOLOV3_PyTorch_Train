import torch
import torch.nn as nn
import torch.nn.functional as F


class ReferenceNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1024):
        super(ReferenceNet, self).__init__()
        self.features = self.make_layers(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        route = None
        for i, feature in enumerate(self.features):
            if isinstance(feature[0], nn.MaxPool2d) and feature[0].stride == 1:
                x = F.pad(x, [0, 1, 0, 1], mode='constant', value=0) # same as paper
            x = feature(x)
            if i == 8:
                route = x
        return x, route

    def make_layers(self, in_channels=3, out_channels=1024):
        # conv: out_channels, kernel_size, stride, batchnorm, activate
        # maxpool: kernel_size stride
        params = [
            [16, 3, 1, True, 'leaky'],
            ['M', 2, 2],
            [32, 3, 1, True, 'leaky'],
            ['M', 2, 2],
            [64, 3, 1, True, 'leaky'],
            ['M', 2, 2],
            [128, 3, 1, True, 'leaky'],
            ['M', 2, 2],
            [256, 3, 1, True, 'leaky'],
            ['M', 2, 2],
            [512, 3, 1, True, 'leaky'],
            ['M', 2, 1], # same as paper
            [out_channels, 3, 1, True, 'leaky'],
        ]

        module_list = nn.ModuleList()
        for i, v in enumerate(params):
            modules = nn.Sequential()
            if v[0] == 'M':
                modules.add_module(f"maxpool_{i}", nn.MaxPool2d(kernel_size=v[1], stride=v[2], padding=int((v[1] - 1) // 2)))
            else:
                modules.add_module(
                    f"conv_{i}",
                    nn.Conv2d(
                        in_channels,
                        v[0],
                        kernel_size=v[1],
                        stride=v[2],
                        padding=(v[1] - 1) // 2,
                        bias=not v[3]
                    )
                )
                if v[3]:
                    modules.add_module(f"bn_{i}", nn.BatchNorm2d(v[0]))
                modules.add_module(f"act_{i}", nn.LeakyReLU(0.1) if v[4] == 'leaky' else nn.ReLU())
                in_channels = v[0]
            module_list.append(modules)
        return module_list