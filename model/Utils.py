import torch
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt, patches


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

    pred_boxes = torch.stack([pred_ctr_x, pred_ctr_y, pred_w, pred_h], dim=2)
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

def center2cordinate(center_size_boxes):
    centers_x = center_size_boxes[:, 0]
    centers_y = center_size_boxes[:, 1]
    widths = center_size_boxes[:, 2]
    heights = center_size_boxes[:, 3]

    x1 = centers_x - 0.5 * widths
    y1 = centers_y - 0.5 * heights
    x2 = centers_x + 0.5 * widths
    y2 = centers_y + 0.5 * heights

    coordinates_boxes = torch.stack([x1, y1, x2, y2], dim=1)
    return coordinates_boxes

def show_image(image, boxes):
    img = np.transpose(image, (1, 2, 0)).numpy()
    plt.imshow(img)
    for box in boxes:
        x, y, w, h = box[0], box[1], box[2], box[3]
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none')
        plt.gca().add_patch(rect)
    plt.axis('off')
    plt.show()

def draw_loss(loss, num_epochs, save_path, title: str = 'Loss Curve Over Epochs', label: str = 'Loss Curve'):
    epochs = range(1, len(num_epochs) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss, label=label)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
