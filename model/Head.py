import torch.nn as nn


class RegressionModel(nn.Module):
    def __init__(self, in_feature_size, num_anchors=9, feature_size=256, bias_init=None):
        """
        Args:
            :param in_feature_size: Number of input channels in the feature map.
            :param num_anchors: Number of anchors used for prediction.
            :param feature_size: Number of channels in intermediate layers.
            :param bias_init: Bias Initializer for the final convolution layer.
        """
        super().__init__()
        self.head = build_head(in_feature_size, num_anchors * 4, bias_init, feature_size)

    def forward(self, x):
        out = self.head(x)

        # out is B x C x W x H, with C = num_anchors * 4
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, in_feature_size, num_anchors=9, num_classes=1, feature_size=256, bias_init=None):
        """
        Args:
            :param in_feature_size: Number of input channels in the feature map.
            :param num_anchors: Number of anchors used for prediction.
            :param num_classes: Number of object classes.
            :param feature_size: Number of channels in intermediate layers.
            :param bias_init: Bias Initializer for the final convolution layer.
        """
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.head = build_head(in_feature_size, self.num_anchors * self.num_classes, bias_init, feature_size)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.head(x)
        out = self.output_act(out)

        # out is B x C x W x H, with C = num_anchors * num_classes
        out = out.permute(0, 2, 3, 1)
        batch_size, width, height, channels = out.shape
        out = out.view(batch_size, width, height, self.num_anchors, self.num_classes)
        return out.contiguous().view(x.shape[0], -1, self.num_classes)


def build_head(in_feature_size, output_filters, bias_init, feature_size=256):
    """Builds predictions 'basehead'.

    Arguments:
        :param in_feature_size: Number of input channels in the feature map.
        :param output_filters: Number of convolution filters in the final layer.
        :param bias_init: Bias Initializer for the final convolution layer.
        :param feature_size: Number of channels in intermediate layers.

    Returns:
        Sequential model representing either the classification
        or the box regression head depending on `output_filters`.

    """
    head = nn.Sequential(
        nn.Conv2d(in_feature_size, feature_size, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(feature_size, output_filters, kernel_size=3, padding=1, bias=bias_init)
    )
    return head


def local_some_test():
    import torch

    in_feature_size = 256
    num_classes = 1
    num_anchors = 9

    classification_head = ClassificationModel(in_feature_size, num_anchors, num_classes)
    box_regression_head = RegressionModel(in_feature_size, num_anchors, num_classes)

    x = torch.randn(1, 256, 16, 25)
    classification_output = classification_head(x)
    box_regression_output = box_regression_head(x)

    print("Output size of the classification head:", classification_output.shape)
    print("Output size of the regression head:", box_regression_output.shape)


def test_with_bias():
    import torch
    import math

    in_feature_size = 256
    num_classes = 1
    num_anchors = 9
    prior_probability = 0.01

    bias_init = nn.init.constant_(torch.empty(1), -math.log((1.0 - prior_probability) / prior_probability))
    classification_head = ClassificationModel(in_feature_size, num_classes=num_classes, bias_init=bias_init)
    box_regression_head = RegressionModel(in_feature_size, bias_init=0)

    x = torch.randn(1, 256, 16, 25)
    classification_output = classification_head(x)
    box_regression_output = box_regression_head(x)

    print("Output size of the classification head using bias:", classification_output.shape)
    print("Output size of the regression head using bias:", box_regression_output.shape)


if __name__ == "__main__":
    local_some_test()
    test_with_bias()
