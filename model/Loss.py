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
        box1_vertices[:, 0] = y_true[:, 0] - 0.5 * y_true[:, 2]  # x1
        box1_vertices[:, 1] = y_true[:, 1] - 0.5 * y_true[:, 3]  # y1
        box1_vertices[:, 2] = y_true[:, 0] + 0.5 * y_true[:, 2]  # x2
        box1_vertices[:, 3] = y_true[:, 1] + 0.5 * y_true[:, 3]  # y2

        iou_matrix = box_iou(box1_vertices, anchor)

        return iou_matrix


class RetinaFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        """
        Inicjalizacja funkcji straty Focal Loss.

        Parametry:
        - alpha: Parametr wpływający na wagę klas pozytywnych i negatywnych.
        - gamma: Parametr regulujący skupienie na trudnych do sklasyfikowania przykładach.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_true, y_pred):
        """
        todo: czy model klasyfikacji daje na wyjściu logity czy prawdopodobieństwa po funkcji aktywacji
        Metoda forward obliczająca Focal Loss.

        Parametry:
        - y_true: Tensor z prawdziwymi etykietami binarnymi (0 lub 1).
        - y_pred: Tensor z przewidywanymi wartościami logits (bez funkcji aktywacji).

        Zwraca:
        - Tensor zawierający wartości Focal Loss dla każdego przykładu.
        """
        cross_entropy = F.cross_entropy(y_pred, y_true, reduction='none')

        # Określenie wag dla klas pozytywnych i negatywnych
        alpha = torch.where(y_true == 1, self.alpha, 1 - self.alpha)

        # Obliczenie pt - czynnika skupiającego uwagę na trudnych przykładach
        pt = torch.where(y_true == 1, y_pred, 1 - y_pred)

        # Obliczenia straty za pomocą Focal Loss
        loss = alpha * (1 - pt)**self.gamma * cross_entropy

        # Sumowanie strat wzdłuż ostatniej osi tensora
        return torch.sum(loss, dim=-1)


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
        self.FocalLoss = RetinaFocalLoss(alpha=alpha, gamma=gamma)
        self.IoU = RetinaNetBoxIoU()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_true, y_classifs, y_regressions, anchors):
        """
        Metoda obliczająca łączną stratę dla modelu RetinaNet.

        Args:
            y_true (torch.Tensor): Tensor zawierający rzeczywiste etykiety.
            y_pred (torch.Tensor): Tensor zawierający przewidywane wartości.

        Returns:
            torch.Tensor: Obliczona łączna strata dla modelu RetinaNet.
        """
        batch_size = y_classifs.shape[0]

        for i in range(batch_size):
            box = y_true[i, :, :4]
            label = y_true[i, :, 4]

            # classif = y_classifs[j, :, :]
            # regression = y_regressions[j, :, :]

            cls_labels = F.one_hot(
                label.long(),
                num_classes=self.num_classes,
            ).float()

            # Obliczenie straty klasyfikacji i iou bounding boxów
            clf_loss = self.FocalLoss(cls_labels, y_classifs) # TODO to na pewno musi byc inaczej bo cls.size to 1,1 a y_classifs to batch (4,153576,1)
            iou_matrix = self.IoU(box, anchors) #TODO box.shape:(1,4) anchors.shape:(1,153576,4)

            IoU_max, IoU_argmax = torch.max(iou_matrix, dim=1)  # num_anchors x 1

            targets = torch.ones(y_classifs.shape) * -1
            targets[torch.lt(IoU_max, 0.4), :] = 0
            positive_idx = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_idx.sum()
            assigned_annotations = box[IoU_argmax, :]

            targets[positive_idx, :] = 0
            targets[positive_idx, assigned_annotations[positive_idx, 4].long()] = 1

            #################
            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() * self.alpha
            else:
                alpha_factor = torch.ones(targets.shape) * self.alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - y_classifs, y_classifs)
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

            bce = -(targets * torch.log(y_classifs) + (1.0 - targets) * torch.log(1.0 - y_classifs))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))

            loss = clf_loss + box_loss
            loss = 0
        return loss


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


def test_focal_loss():
    y_true = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
    y_pred = torch.tensor([0.2, -0.5, 1.2, 0.8], dtype=torch.float32)
    focal = RetinaFocalLoss()
    loss_value_single_class = focal(y_true, y_pred)
    print("Focal Loss (Single Class):", loss_value_single_class.item())


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
    test_focal_loss()
    test_retina_loss()
