import torch.nn as nn
import math
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from model.FeaturesPyramid import FeaturesPyramid
from model.ResNet50 import ResNet50
from model.Head import ClassificationModel, RegressionModel
from model.Anchors import Anchors
from model.Loss import RetinaNetLoss
from model.Utils import regression2BoxTransform, trimBox2Image, center2cordinate, show_image
from torchvision.ops import nms

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

        bias_init_cls = -math.log((1.0 - self.prior_probability) / self.prior_probability)
        self.ClassificationModel = ClassificationModel(self.fpn_feature_size, num_classes=num_classes,
                                                       bias_init=bias_init_cls)
        self.RegressionModel = RegressionModel(self.fpn_feature_size, bias_init=0)

        self.anchors = Anchors()

        self.FocalLoss = RetinaNetLoss()

    def forward(self, x):
        '''
        :param x: input tensor
        :return:
        '''
        if self.training or self.eval():
            img, annotations = x
        else:
            img = x

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        c3, c4, c5 = self.Backbone(img)
        fpn_features_map = self.FPN([c3, c4, c5])

        # concatenation for all feature levels [p3,p4,p5,p6,p7]
        regression = torch.cat([self.RegressionModel(feature) for feature in fpn_features_map], dim=1)

        classification = torch.cat([self.ClassificationModel(feature) for feature in fpn_features_map], dim=1)

        anchors = self.anchors(img)

        if self.training:
            return self.FocalLoss(annotations, classification, regression, anchors)
        else:
            transformed_anchors = regression2BoxTransform(anchors, regression)
            transformed_anchors = trimBox2Image(transformed_anchors, img)

            finalScores = torch.empty(0, device=device)
            finalAnchorBoxesLabels = torch.empty(0, device=device).long()
            if torch.cuda.is_available():
                finalAnchorBoxesCoordinates = torch.Tensor([]).cuda()
            else:
                finalAnchorBoxesCoordinates = torch.Tensor([])

            for i in range(classification.shape[2]):
                scores = torch.squeeze(classification[:, :, i])
                scores_over_thresh = (scores > 0.1)
                if scores_over_thresh.sum() == 0:
                    continue    # no boxes to NMS, just continue

                # NMS - Non maximum suppression - usuwanie obszarów, które mają mniejsze prawdopodobieństwo obecności
                # obiektu lub są silnie pokrywajace się z innymi obszarami o wyższym prawdopodobieństwie
                scores = scores[scores_over_thresh]
                anchorBoxes = torch.squeeze(transformed_anchors)[scores_over_thresh]

                coordinates_anchorBoxes = center2cordinate(anchorBoxes)

                anchors_nms_idx = nms(coordinates_anchorBoxes, scores, 0.5)

                finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                finalAnchorBoxesLabels = torch.cat((finalAnchorBoxesLabels, torch.tensor([i] * anchors_nms_idx.shape[0], device=device)))
                finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))
            return [finalScores, finalAnchorBoxesLabels, finalAnchorBoxesCoordinates]


def evaluate(dataset, model, threshold=0.05):
    model.eval()
    with torch.no_grad():
        results = []
        true = []
        images_info = []
        counter = 0
        for index in range(len(dataset)):
            data = dataset[index]

            if torch.cuda.is_available():
                scores, labels, boxes = model([data['img'].cuda().float().unsqueeze(0), data['boxes_list'].cuda().unsqueeze(0)])
            else:
                scores, labels, boxes = model([data['img'].float().unsqueeze(0), data['boxes_list'].unsqueeze(0)])
            scores = scores.cpu()
            labels = labels.cpu()
            boxes = boxes.cpu()

            if boxes.shape[0] > 0:
                # compute predicted labels and scores
                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]

                    # scores are sorted, so we can break
                    if score < threshold:
                        break

                    # append detection for each positively labeled class
                    image_result = {
                        'image_id': index,
                        'category_id': 1, #jest tylko jedna klasa
                        'score': float(score),
                        'bbox': box.tolist(),
                        'iscrowd': 0,
                    }
                    results.append(image_result)

                # show_image(data['img'], boxes)


            for i in range(data['boxes_num']):
                image_result = {
                    'id': counter,
                    'image_id': index,
                    'category_id': 1,  # jest tylko jedna klasa
                    'area': data['boxes_list'][i, 2] * data['boxes_list'][i, 3],
                    'bbox': data['boxes_list'][i, :4].float().tolist(),
                    'iscrowd': 0,
                }
                counter += 1
                true.append(image_result)

            image_info = {
                'id': index,
                'width': data['img'].shape[2],
                'height': data['img'].shape[2],
            }
            images_info.append(image_info)

            # print progress
            print('{}/{}'.format(index, len(dataset)), end='\r')
            #if index == 2: break

        categories_info = [
            {'id': 1, 'name': 'twarz'},
        ]

        coco_gt = COCO()  # Utwórz instancję COCO dla prawdziwych bounding boxów
        coco_gt.dataset['annotations'] = true
        coco_gt.dataset['images'] = images_info
        coco_gt.dataset['categories'] = categories_info
        coco_gt.createIndex()

        coco_dt = coco_gt.loadRes(results)  # Utwórz instancję COCO dla predykowanych bounding boxów

        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return




from WiderDataLoader.wider_loader import WiderFaceDataset
from WiderDataLoader.wider_batch_iterator import BatchIterator
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, ToPILImage, Resize

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

    print("Bias klasyfikacji:", RetNet.ClassificationModel.head[-1].bias.data)
    print("Bias regresji:", RetNet.RegressionModel.head[-1].bias.data)

def test_eval():
    DATA_DIR = '../WIDER'
    transform = Compose(
        [ToPILImage(), Resize((800, 1024)), ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    wider_val_dataset = WiderFaceDataset(DATA_DIR, split='val', transform=transform)
    model = RetinaNet()
    evaluate(wider_val_dataset, model=model, threshold=0.05)


if __name__ == "__main__":
    #test_output_from_head()
    test_eval()

