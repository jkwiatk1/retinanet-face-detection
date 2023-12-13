import torch
import numpy as np
import torch.nn as nn


def regression2BoxTransform(anchors, regression, std=None):
    """
         Przekształć predykcje na faktyczne rozmiary boxow

         Parametry:
         - anchors: oryginalne prostokątne obszary (tensor)
         - regression: predykcje dla prostokątnych obszarów (tensor)
         - std: odwrotna operacja normalizacji danych regresji

         Zwraca:
         - predykcje prostokątnych obszarów po przekształceniu (tensor)
    """
    if std is None:
        if torch.cuda.is_available():
            std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
        else:
            std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))

    # Przeliczanie na x, y, width, height
    widths = anchors[:, :, 2] - anchors[:, :, 0]
    heights = anchors[:, :, 3] - anchors[:, :, 1]
    ctr_x = anchors[:, :, 0] + 0.5 * widths
    ctr_y = anchors[:, :, 1] + 0.5 * heights

    # Operacja odwrotna niż podczas uczenia w funkcji Loss
    dx = regression[:, :, 0] * std[0]  # + self.mean[0]
    dy = regression[:, :, 1] * std[1]  # + self.mean[1]
    dw = regression[:, :, 2] * std[2]  # + self.mean[2]
    dh = regression[:, :, 3] * std[3]  # + self.mean[3]

    # Obliczenie predykcji prostokątnych obszarów
    pred_ctr_x = ctr_x + dx * widths
    pred_ctr_y = ctr_y + dy * heights
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    # Przeliczenie na x1,y1,x2,y2
    pred_x1 = pred_ctr_x - 0.5 * pred_w
    pred_y1 = pred_ctr_y - 0.5 * pred_h
    pred_x2 = pred_ctr_x + 0.5 * pred_w
    pred_y2 = pred_ctr_y + 0.5 * pred_h

    pred_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=2)
    return pred_boxes


def trimBox2Image(boxes, img):
    """
        Przycina prostokątne obszary do granic obrazu.

        Parametry:
        - boxes: prostokątne obszary do przycięcia (tensor)
        - img: obraz, do którego należy dopasować granice (tensor)

        Zwraca:
        - przycięte prostokątne obszary (tensor)
    """
    # Pobranie wymiarów obrazu
    batch_size, num_channels, height, width = img.shape

    # Przycięcie prostokątnych obszarów do granic obrazu
    boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
    boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)
    boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
    boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

    return boxes
