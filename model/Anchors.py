import numpy as np
import torch
import torch.nn as nn


class Anchors(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        """
        Moduł Anchorów służący do generowania pudełek zakotwiczenia używanych w modelach detekcji obiektów.

        Argumenty:
        - pyramid_levels (lista): Lista liczb reprezentujących poziomy piramidy.
        - strides (lista): Lista liczb reprezentujących przesunięcia anchor box dla każdego poziomu piramidy.
        - sizes (lista): Lista liczb reprezentujących podstawowe rozmiary pudełek na każdym poziomie piramidy.
        - ratios (numpy array): Tablica NumPy reprezentująca stosunki wysokości do szerokości pudełka
        - scales (numpy array): Tablica NumPy reprezentująca skale pudełek zakotwiczenia.

        Jeśli którykolwiek z argumentów nie zostanie dostarczony, zostaną użyte wartości domyślne.
        """

        super(Anchors, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        if sizes is None:
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])
        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    def forward(self, image):
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 4)).astype(np.float32)

        for idx, p in enumerate(self.pyramid_levels):
            anchors = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

        all_anchors = np.expand_dims(all_anchors, axis=0)

        if torch.cuda.is_available():
            return torch.from_numpy(all_anchors.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchors.astype(np.float32))


def generate_anchor_boxes(base_size=256, ratios=None, scales=None):
    """
    Generuje anchor boxy dla Region Proposal Network (RPN).

    Parameters:
    - base_size: Rozmiar bazowy anchor boxa.
    - ratios: Stosunek wysokości do szerokości.
    - scales: Lista skal anchor boxów.

    Returns:
    - anchor_boxes: NumPy array zawierający współrzędne anchor boxów w formacie (x_min, y_min, x_max, y_max).
    """
    if ratios is None:
        ratios = np.array([0.5, 1, 2])
    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    anchor_boxes = []
    for ratio in ratios:
        for scale in scales:
            width = base_size * scale * np.sqrt(ratio)
            height = base_size * scale / np.sqrt(ratio)
            anchor_box = [-width / 2, -height / 2, width / 2, height / 2]
            anchor_boxes.append(anchor_box)
    return np.array(anchor_boxes)


def compute_shape(image_shape, pyramid_levels):
    """Compute shapes based on pyramid levels.

    :param image_shape:
    :param pyramid_levels:
    :return:
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes




#print(generate_anchor_boxes(16))
#print(compute_shape([500, 500, 3], [1, 2, 3, 4, 5]))

