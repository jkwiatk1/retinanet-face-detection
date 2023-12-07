import torch.nn as nn
from model.FeaturesPyramid import FeaturesPyramid
from model.ResNet50 import ResNet50


class RetinaNet(nn.Module):
    def __init__(self, backbone='resnet50'):
        super().__init__()

        self.fpn_feature_size = 256

        if backbone == 'resnet50':
            self.Backbone = ResNet50()
            self.in_channels_size_list = [512, 1024, 2048]
        else:
            raise NotImplementedError(f"This backbone ({backbone}) is not implemented.")

        self.FPN = FeaturesPyramid(self.in_channels_size_list, out_channels=self.fpn_feature_size)

    def _make_prediction_layer(self):
        pass

    def forward(self, x):
        '''
        :param x: input tensor
        :return:
        '''
        c3, c4, c5 = self.Backbone(x)
        p3, p4, p5, p6, p7 = self.FPN([c3, c4, c5])
        return [p3, p4, p5, p6, p7]



def local_some_test():
    import torch
    RetNet = RetinaNet()
    input_tensor = torch.randn(1, 3, 512, 800)
    print(input_tensor.size())
    outputs_RN = RetNet(input_tensor)

    for i, output in enumerate(outputs_RN):
        print(f'Output P{i + 3} shape: {output.shape}')

if __name__ == "__main__":
    local_some_test()

