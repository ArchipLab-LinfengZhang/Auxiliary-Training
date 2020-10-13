import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': './pretrain/resnet18-5c106cde.pth',
    'resnet34': './pretrain/resnet34-333f7ec4.pth',
    'resnet50': './pretrain/resnet50-19c8e357.pth',
    'resnet101': './pretrain/resnet101-5d3b4d8f.pth',
    'resnet152': './pretrain/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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


class MainClassifier(nn.Module):
    def __init__(self, channel, num_classes=100):
        super(MainClassifier, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channel, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class AuxiliaryClassifier(nn.Module):
    #   Auxiliary classifier, including first an attention layer, then a bottlecneck layer,
    #   and final a fully connected layer

    def __init__(self, channel, num_classes=100):
        super(AuxiliaryClassifier, self).__init__()
        self.attention_layer = self._make_attention_layer(channel)
        self.bottleneck_layer = self._make_bottleneck_layer(channel)
        self.fc = nn.Linear(channel, num_classes)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def _make_attention_layer(self, channel):
        #   parameter <stride=1> indicates the stride of bottleneck layers
        #   for CIFAR datasets, stride=1 is advised
        return nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(kernel_size=3, padding=1, stride=1,
                               in_channels=channel, out_channels=channel, bias=False),
            nn.BatchNorm2d(channel),
            nn.Sigmoid()
        )

    def _make_bottleneck_layer(self, channel):
        return nn.Sequential(
            nn.Conv2d(channel, channel // 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 2, channel // 2, kernel_size=3, stride=1),
            nn.BatchNorm2d(channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 2, channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        x = self.bottleneck_layer(self.attention_layer(x) * x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100, zero_init_residual=False):
        super(ResNet, self).__init__()
        print("num_class: ", num_classes)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        #   self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #   remove maxplooing in cifar datasets

        self.layer1 = self._make_layer(block,  64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.bn_means, self.bn_vars = [], []
        self.auxiliary_classifiers_list = nn.ModuleList([
            MainClassifier(channel=512 * block.expansion, num_classes=num_classes),
            AuxiliaryClassifier(channel=512 * block.expansion, num_classes=num_classes),
            AuxiliaryClassifier(channel=512 * block.expansion, num_classes=num_classes),
            AuxiliaryClassifier(channel=512 * block.expansion, num_classes=num_classes),
            AuxiliaryClassifier(channel=512 * block.expansion, num_classes=num_classes)
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def close_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False

    def record_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.bn_means.append(m.running_mean.clone())
                self.bn_vars.append(m.running_var.clone())

    def open_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = True

    def reset_bn(self):
        self.bn_means = []
        self.bn_vars = []

    def load_bn(self):
        index = 0
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.data = self.bn_means[index].clone()
                m.running_var.data = self.bn_vars[index].clone()
                #   self.bn_means.append(m.running_mean)
                #   self.bn_vars.append(m.running_var)
                index += 1
        self.bn_vars = []
        self.bn_means = []

    def print_bn(self):
        for m in self.layer1.modules():
            if isinstance(m, nn.BatchNorm2d):
                print(m.running_var[0])
                print(m.running_mean[0])

    def forward(self, inputs):
        out_list = []
        for index in range(len(inputs)):
            #   print(index)
            x = inputs[index]
            out = self.conv1(x)
            out = self.bn1(out)
            layer1_out = self.layer1(out)
            layer2_out = self.layer2(layer1_out)
            layer3_out = self.layer3(layer2_out)
            layer4_out = self.layer4(layer3_out)
            out_list.append(self.auxiliary_classifiers_list[index](layer4_out))
            if index == 0:
                #return layer4_out
                self.record_bn()
        self.load_bn()
        return out_list


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))

    return model
