import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou


class RetinaNetBoxIoU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, anchor):
        """
        Metoda forward obliczająca wartosc iou

        Parametry:
        - y_true: Tensor z prawdziwymi boxami (ground truth), zawierającymi informacje o położeniu obiektów.
                    (x_srodek, y_srodek, szerokosc, wysokosc)
        - anchor: Anchor box

        Zwraca:
        - wartosc iou
        """

        box1_vertices = torch.zeros_like(y_true)
        if torch.cuda.is_available():
            box1_vertices = box1_vertices.cuda()

        box1_vertices[:, 0] = y_true[:, 0] - 0.5 * y_true[:, 2]  # x1
        box1_vertices[:, 1] = y_true[:, 1] - 0.5 * y_true[:, 3]  # y1
        box1_vertices[:, 2] = y_true[:, 0] + 0.5 * y_true[:, 2]  # x2
        box1_vertices[:, 3] = y_true[:, 1] + 0.5 * y_true[:, 3]  # y2

        iou_matrix = box_iou(box1_vertices, anchor)

        return iou_matrix

'''This class is based on code from https://github.com/yhenon/pytorch-retinanet'''
class RetinaNetLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=1):
        """
        Inicjalizacja klasy RetinaNetLoss.

        Args:
            num_classes (int): Liczba klas w zadaniu detekcji obiektów.
            alpha (float): Parametr kontrolujący wagę klas pozytywnych i negatywnych w klasyfikacji.
            gamma (float): Parametr gamma używany w funkcji Focal Loss.
        """
        super(RetinaNetLoss, self).__init__()
        self.IoU = RetinaNetBoxIoU()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_true_tmp, y_classifs, y_regressions, anchors):
        """
        Metoda obliczająca łączną stratę dla modelu RetinaNet.

        Args:
            y_true (torch.Tensor): Tensor zawierający rzeczywiste etykiety.
            y_pred (torch.Tensor): Tensor zawierający przewidywane wartości.

        Returns:
            torch.Tensor: Obliczona łączna strata dla modelu RetinaNet.
        """
        classification_losses = []
        regression_losses = []
        batch_size = y_classifs.shape[0]

        anchor = anchors[0, :, :]
        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

        # Class loss
        for i in range(batch_size):
            box_annotation = y_true_tmp[i,:,:]
            y_true = box_annotation[box_annotation[:, 4] != -1]
            true_boxes = y_true[:, :4]

            predict_labels = y_classifs[i, :, :]
            predict_boxes = y_regressions[i, :, :]

            if true_boxes.shape[0] == 0:
                if torch.cuda.is_available():
                    alpha_factor = torch.ones(predict_labels.shape, device='cuda') * self.alpha
                else:
                    alpha_factor = torch.ones(predict_labels.shape) * self.alpha
                alpha_factor = 1. - alpha_factor
                focal_weight = predict_labels
                focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)
                bce = -(torch.log(1.0 - predict_labels))
                cls_loss = focal_weight * bce
                classification_losses.append(cls_loss.sum())

                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())
                continue



            # Obliczenie straty klasyfikacji i iou bounding boxów
            iou_matrix = self.IoU(true_boxes, anchor).t()  # true_boxes.shape:(1,4) anchors.shape:(153576,4)
            IoU_max, IoU_argmax = torch.max(iou_matrix, dim=1)  # num_anchors x 1

            if torch.cuda.is_available():
                targets = torch.ones(predict_labels.shape, device='cuda') * -1
            else:
                targets = torch.ones(predict_labels.shape) * -1

            targets[torch.lt(IoU_max, 0.4), :] = 0
            positive_idx = torch.ge(IoU_max, 0.5)
            num_positive_anchors = positive_idx.sum()
            assigned_annotations = y_true[IoU_argmax, :]
            targets[positive_idx, :] = 0
            targets[positive_idx, assigned_annotations[positive_idx, 4].long()] = 1

            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape, device='cuda') * self.alpha
            else:
                alpha_factor = torch.ones(targets.shape) * self.alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - predict_labels, predict_labels)
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

            bce = -(targets * torch.log(predict_labels) + (1.0 - targets) * torch.log(1.0 - predict_labels))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape, device='cuda'))
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))


            # Regression loss
            if positive_idx.sum() > 0:
                assigned_annotations = assigned_annotations[positive_idx, :]

                anchor_widths_pi = anchor_widths[positive_idx]
                anchor_heights_pi = anchor_heights[positive_idx]
                anchor_ctr_x_pi = anchor_ctr_x[positive_idx]
                anchor_ctr_y_pi = anchor_ctr_y[positive_idx]

                gt_ctr_x = assigned_annotations[:, 0]
                gt_ctr_y = assigned_annotations[:, 1]
                gt_widths = torch.clamp(assigned_annotations[:, 2], min=1)
                gt_heights = torch.clamp(assigned_annotations[:, 3], min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                if torch.cuda.is_available():
                    targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                else:
                    targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

                y_regression_diff = torch.abs(targets - predict_boxes[positive_idx, :])
                regression_loss = torch.where(
                    torch.le(y_regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(y_regression_diff, 2),
                    y_regression_diff - 0.5 / 9.0
                )

                regression_losses.append(regression_loss.mean())

            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)


def test_box_iou():
    import pytest
    y_true = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.5, 0.5, 1.0, 1.0]])
    anchor = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.5, 0.5, 1.0, 1.0]])

    retinanet_box_iou = RetinaNetBoxIoU()

    iou_matrix = retinanet_box_iou(y_true, anchor)

    assert isinstance(iou_matrix, torch.Tensor)

    assert iou_matrix.shape == (2, 2)

    # assert iou_matrix[0, 0].item() == pytest.approx(1.0)
    # assert iou_matrix[0, 1].item() == pytest.approx(0.25)
    # assert iou_matrix[1, 0].item() == pytest.approx(0.25)
    # assert iou_matrix[1, 1].item() == pytest.approx(1.0)

    print("\nGround Truth:")
    print(y_true)
    print("\nAnchor:")
    print(anchor)
    print("\nIoU:")
    print(iou_matrix)


def test_retina_loss():
    num_classes = 1
    alpha = 0.25
    gamma = 2.0
    delta = 1.0
    retina_loss = RetinaNetLoss(num_classes=num_classes, alpha=alpha, gamma=gamma, delta=delta)
    # Rzeczywiste etykiety (tensor o wymiarach [batch_size, max_boxes, 5]), klasa 0 lub 1
    y_true = torch.tensor([
        [[10, 20, 50, 70, 1], [30, 40, 80, 90, -1]],
        [[15, 25, 55, 75, 1], [35, 45, 85, 95, 0]],
    ])
    # Przewidywane (tensor o wymiarach [batch_size, max_boxes, 5 + num_classes]), przedostatni - prawd. braku klasy
    y_pred = torch.tensor([
        [[9, 19, 48, 68, 0.9, 0.1], [29, 39, 78, 88, 0.2, 0.8]],
        [[14, 24, 54, 74, 0.95, 0.05], [34, 44, 84, 94, 0.3, 0.7]],
    ])
    loss = retina_loss(y_true, y_pred)
    print("Łączna strata:", loss)


if __name__ == "__main__":
    test_box_iou()
    test_retina_loss()
