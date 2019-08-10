import torch.nn as nn
from torch.autograd import Function
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math
from misc import init_weights

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


class DomainClassifier(nn.Module):
    def __init__(self, input_dimension):
        super(DomainClassifier, self).__init__()
        self.domain_classifier = nn.Sequential(
            nn.Linear(input_dimension, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.domain_classifier(x)


class ResenetDomainClassifier(nn.Module):
    def __init__(self, input_dimension):
        super(ResenetDomainClassifier, self).__init__()
        self.domain_classifier = nn.Sequential(
            nn.Linear(input_dimension, 1),
            nn.Sigmoid()
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.domain_classifier(x)


class MADA_USPSandMNIST(nn.Module):
    def __init__(self):
        super(MADA_USPSandMNIST, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, (5, 5))
        self.conv1_1_bn = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2_1 = nn.Conv2d(32, 64, (3, 3))
        self.conv2_1_bn = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, (3, 3))
        self.conv2_2_bn = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.drop1 = nn.Dropout()

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(1024, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))

        self.domain_classifiers = nn.ModuleList([
            DomainClassifier(1024) for _ in range(10)
        ])

    def forward(self, input_data, alpha):
        x = self.pool1(F.relu(self.conv1_1_bn(self.conv1_1(input_data))))
        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        x = self.pool2(F.relu(self.conv2_2_bn(self.conv2_2(x))))
        x = x.view(-1, 1024)
        x = self.drop1(x)

        class_output = self.class_classifier(x)
        reverse_feature = ReverseLayerF.apply(x, alpha)
        predictions = F.softmax(class_output, dim=1).detach()
        domain_logits = []
        class_idx = 0
        for domain_classifier in self.domain_classifiers:
            weighted_features = predictions[:, class_idx].unsqueeze(1) * reverse_feature
            domain_logits.append(domain_classifier(weighted_features))
            class_idx += 1
        return class_output, domain_logits


class MADA_SVHN(nn.Module):
    def __init__(self):
        super(MADA_SVHN, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 128, (3, 3), padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(128)
        self.conv1_2 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(128)
        self.conv1_3 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv1_3_bn = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.drop1 = nn.Dropout()

        self.conv2_1 = nn.Conv2d(128, 256, (3, 3), padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(256)
        self.conv2_2 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(256)
        self.conv2_3 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv2_3_bn = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.drop2 = nn.Dropout()

        self.conv3_1 = nn.Conv2d(256, 512, (3, 3), padding=0)
        self.conv3_1_bn = nn.BatchNorm2d(512)
        self.nin3_2 = nn.Conv2d(512, 256, (1, 1), padding=1)
        self.nin3_2_bn = nn.BatchNorm2d(256)
        self.nin3_3 = nn.Conv2d(256, 128, (1, 1), padding=1)
        self.nin3_3_bn = nn.BatchNorm2d(128)

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(128, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))

        self.domain_classifiers = nn.ModuleList([
            DomainClassifier(128) for _ in range(10)
        ])

    def forward(self, input_data, alpha):
        x = F.relu(self.conv1_1_bn(self.conv1_1(input_data)))
        x = F.relu(self.conv1_2_bn(self.conv1_2(x)))
        x = self.pool1(F.relu(self.conv1_3_bn(self.conv1_3(x))))
        x = self.drop1(x)

        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        x = F.relu(self.conv2_2_bn(self.conv2_2(x)))
        x = self.pool2(F.relu(self.conv2_3_bn(self.conv2_3(x))))
        x = self.drop2(x)

        x = F.relu(self.conv3_1_bn(self.conv3_1(x)))
        x = F.relu(self.nin3_2_bn(self.nin3_2(x)))
        x = F.relu(self.nin3_3_bn(self.nin3_3(x)))

        x = F.avg_pool2d(x, 6)
        x = x.view(-1, 128)
        class_output = self.class_classifier(x)
        reverse_feature = ReverseLayerF.apply(x, alpha)
        predictions = F.softmax(class_output, dim=1).detach()
        domain_logits = []
        class_idx = 0
        for domain_classifier in self.domain_classifiers:
            weighted_features = predictions[:, class_idx].unsqueeze(1) * reverse_feature
            domain_logits.append(domain_classifier(weighted_features))
            class_idx += 1
        return class_output, domain_logits


def load_model(category, device='cpu'):
    if category == 'utom' or category == 'mtou':
        return MADA_USPSandMNIST(device)
    elif category == 'stom':
        return MADA_SVHN()


def load_MADA(num_class=31):
    netG = MADA(num_class)
    url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    pretrained_dict = model_zoo.load_url(url)
    model_dict = netG.state_dict()
    for k, v in model_dict.items():
        if "class_classifier" not in k and 'domain_classifier' not in k and 'num_batches_tracked' not in k and 'bottleneck'not in k:
            model_dict[k] = pretrained_dict[k[k.find(".") + 1:]]
    netG.load_state_dict(model_dict)
    return netG


class MADA(nn.Module):
    def __init__(self, num_classes=31):
        super(MADA, self).__init__()
        self.feature = resnet50(False)

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc3', nn.Linear(2048, num_classes))
        self.class_classifier.apply(init_weights)

        self.domain_classifiers = nn.ModuleList([
            ResenetDomainClassifier(2048) for _ in range(num_classes)
        ])

    def forward(self, input_data, alpha):
        feature = self.feature(input_data)
        feature = feature.view(-1, 2048)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        predictions = F.softmax(class_output, dim=1).detach()

        domain_logits = []
        class_idx = 0
        for domain_classifier in self.domain_classifiers:
            weighted_features = predictions[:, class_idx].unsqueeze(1) * reverse_feature
            domain_logits.append(domain_classifier(weighted_features))
            class_idx += 1
        return class_output, domain_logits


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.baselayer = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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
        # print(x.size())
        x = x.view(x.size(0), -1)

        return x


def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


if __name__ == '__main__':
    netG = MADA()
    # url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    # pretrained_dict = model_zoo.load_url(url)
    model_dict = netG.state_dict()
    for k, v in model_dict.items():
        print(k)
    #     if "class_classifier" not in k and 'domain_classifier' not in k and 'num_batches_tracked' not in k:
    #         model_dict[k] = pretrained_dict[k[k.find(".") + 1:]]
    # netG.load_state_dict(model_dict)
    # print(model_dict)
    # tensor = t
    # torch.Tensor(64, 3, 224, 224)
    # out1, out2 = net(tensor, 0.1)
    # print(out1.shape)
    # print(out2.shape)
    # t = torch.Tensor(64, 3, 32, 32)
    # net = DANN_SVHN()
    # out1, out2 = net(t, 0)
    # print(out1.size())
    # print(out2.size())
