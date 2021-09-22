import torch.nn as nn
import networks.layers_WS as L

__all__ = ['ResNet', 'l_resnet50']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return L.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return L.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, custom_groupnorm=False):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = L.norm(planes, custom_groupnorm)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = L.norm(planes, custom_groupnorm)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = L.norm(planes * self.expansion, custom_groupnorm)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=1000, custom_groupnorm=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = L.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                              bias=False)
        self.bn1 = L.norm(64, custom_groupnorm)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        self.layer1 = self._make_layer(block, 64, layers[0], custom_groupnorm=custom_groupnorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, custom_groupnorm=custom_groupnorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,custom_groupnorm=custom_groupnorm)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,custom_groupnorm=custom_groupnorm)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, custom_groupnorm=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                L.norm(planes * block.expansion, custom_groupnorm),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, custom_groupnorm=custom_groupnorm))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, custom_groupnorm=custom_groupnorm))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
