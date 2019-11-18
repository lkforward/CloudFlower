#
import torch
import torch.nn as nn


class basicBlock(nn.Module):
    def __init__(self):
        self.conv1 = nn.Module.conv2d()
        # NOTE: Relu as the activation?
        self.activation = None
        self.conv2 = nn.Module.conv2d()

        pass

    def forward(self, x):
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)

        out = self.activation(out + x)

        return out

# Similarly, we can define bottleneck block:

class resNet(nn.Module):
    def __init__(self, layers):
        self.layers = layers
        assert(len(layers) == 5), "layers should have a length of 5!"

        return self

    def _make_layer(self, n_blocks):
        layer = nn.ModuleList()

        for _ in range(n_blocks):
            layer.append(basicBlock())

        return layer

    def forward(self, x):


        layer1 = self._make_layer(self.layers[0])
        layer2 = self._make_layer(self.layers[1])
        layer3 = self._make_layer(self.layers[2])
        layer4 = self._make_layer(self.layers[3])
        layer5 = self._make_layer(self.layers[4])

        x = layer1(x)
        x = layer2(x)
        x = layer3(x)
        x = layer4(x)
        x = layer5(x)

        return x
