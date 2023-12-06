import torch.nn as nn
from model.FeaturesPyramid import FeaturesPyramid
from model.ResNet50 import ResNet50


class RetinaNet(nn.Module):
    def __init__(self, backbone='resnet50'):
        super().__init__()

        self.fpn_feature_size = 256

        if backbone == 'resnet50':
            self.backbone = ResNet50()
            self.in_channels_size_list = [512, 1024, 2048]
        else:
            raise NotImplementedError(f"This backbone ({backbone}) is not implemented.")

        self.fpn = FeaturesPyramid(self.in_channels_size_list, out_channels=self.fpn_feature_size)

    def _make_prediction_layer(self):
        pass

    def forward(self, x):
        pass
