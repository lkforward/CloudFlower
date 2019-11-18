#
import torch
import torch.nn as nn

# We have had the basic structure of the code.
# Although some details needs to be refined.

# To use the resnet model in pytorch:
#   model_conv=torchvision.models.resnet50(pretrained=False)
# (Ref: https://missinglink.ai/guides/pytorch/pytorch-resnet-building-training-scaling-residual-networks-pytorch/)
#
# For more parameter settings in resnet, we need to check the parameters in the ResNet class:
# https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html.

class basicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        # Both layers are 3 by 3 conv; but the in/out channels are changing with layer.
        self.conv1 = nn.Module.Conv2d(in_channels=inplanes, out_channels=planes,
                                      kernel_size=3,
                                      stride=1, padding=0, dilation=1,
                                      groups=1, bias=False)
        self.conv2 = nn.Module.Conv2d(in_channels=planes, out_channels=planes,
                                      kernel_size=3,
                                      stride=1, padding=0, dilation=1,
                                      groups=1, bias=False)
        self.activation = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)

        pass

    def forward(self, x):
        out = self.conv1(x)
        out = self.activation(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out += x
        out = self.activation(out)
        out = self.bn2(out)

        return out

# Similarly, we can define bottleneck block:

class resNet(nn.Module):
    def __init__(self, layers):
        self.layers = layers
        assert(len(layers) == 4), "ResNet50 and ResNet34 should have a length of 4!"

        return self

    def _make_layer(self, n_blocks):
        layer = nn.ModuleList()

        for _ in range(n_blocks):
            layer.append(basicBlock(inplanes=0, planes=0))

        return layer

    def forward(self, x):
        # A conv2d layer as the first layer:

        # Four grouped layers of resnet
        layer1 = self._make_layer(self.layers[0])
        layer2 = self._make_layer(self.layers[1])
        layer3 = self._make_layer(self.layers[2])
        layer4 = self._make_layer(self.layers[3])

        x = layer1(x)
        x = layer2(x)
        x = layer3(x)
        x = layer4(x)

        # Finally a FC layer
        return x
