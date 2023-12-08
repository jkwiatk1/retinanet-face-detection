import torch.nn as nn
import math
import torch
from model.FeaturesPyramid import FeaturesPyramid
from model.ResNet50 import ResNet50
from model.Head import ClassificationModel, RegressionModel


class RetinaNet(nn.Module):
    def __init__(self, num_classes=1, backbone='resnet50'):
        super().__init__()

        self.fpn_feature_size = 256
        self.prior_probability = 0.01

        if backbone == 'resnet50':
            self.Backbone = ResNet50()
            self.in_channels_size_list = [512, 1024, 2048]
        else:
            raise NotImplementedError(f"This backbone ({backbone}) is not implemented.")

        self.FPN = FeaturesPyramid(self.in_channels_size_list, out_channels=self.fpn_feature_size)

        #TODO Init bias in heads  
        bias_init = nn.init.constant_(torch.empty(1),
                                      -math.log((1.0 - self.prior_probability) / self.prior_probability))
        self.ClassificationModel = ClassificationModel(self.fpn_feature_size, num_classes=num_classes,
                                                       bias_init=None)
        self.RegressionModel = RegressionModel(self.fpn_feature_size, bias_init=None)

    def forward(self, x):
        '''
        :param x: input tensor
        :return:
        '''
        c3, c4, c5 = self.Backbone(x)
        fpn_features_map = self.FPN([c3, c4, c5])

        # concatenation for all feature levels
        regression = torch.cat([self.RegressionModel(feature) for feature in fpn_features_map], dim=1)

        classification = torch.cat([self.ClassificationModel(feature) for feature in fpn_features_map], dim=1)

        return regression, classification


def test_output_from_fpn():
    import torch
    RetNet = RetinaNet()
    input_tensor = torch.randn(1, 3, 512, 800)
    print(input_tensor.size())
    outputs_RN = RetNet(input_tensor)

    for i, output in enumerate(outputs_RN):
        print(f'Output P{i + 3} shape: {output.shape}')


def test_output_from_head():
    import torch
    RetNet = RetinaNet()
    input_tensor = torch.randn(1, 3, 512, 800)
    print("Input tensor size:")
    print(input_tensor.size())
    output_regression_RN, output_class_RN = RetNet(input_tensor)

    print("Output size of the classification head:", output_class_RN.shape)
    print("Output size of the regression head:", output_regression_RN.shape)


if __name__ == "__main__":
    test_output_from_head()
