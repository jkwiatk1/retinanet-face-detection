import numpy as np
import torch
import torch.nn as nn
import numpy as np


'''This class is based on code from https://github.com/yhenon/pytorch-retinanet'''
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
            self.sizes = [2 ** (x + 1) for x in self.pyramid_levels]
        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])
        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    def forward(self, image):
        shape = image.shape[2:]
        shape = np.array(shape)
        image_shapes = [(shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        all_anchors = np.zeros((0, 4)).astype(np.float32)
        for idx, p in enumerate(self.pyramid_levels):
            anchors = generate_anchor_boxes(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
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
            anchor_box = [-height / 2, -width / 2, height / 2, width / 2]
            anchor_boxes.append(anchor_box)
    return np.array(anchor_boxes)


def shift(shape, stride, anchors):
    """
    Przesuwa kotwice na siatkę punktów o określonym kroku.

    Args:
    - shape (tuple): Krotka zawierająca wysokość i szerokość siatki przesunięć
    - stride (int): Kroki przesunięć na osiach X i Y dla generowania siatki punktów.
    - anchors (numpy array): Tablica NumPy reprezentująca kotwice, czyli początkowe pudełka zakotwiczenia.

    Returns:
    - numpy array: Tablica NumPy reprezentująca wszystkie przesunięte kotwice na siatkę punktów.
    """
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride
    x, y = np.meshgrid(shift_x, shift_y)
    x = x.ravel()
    y = y.ravel()
    shifts = np.vstack((x, y, x, y)).transpose()
    all_anchors = anchors.reshape((1, -1, 4)) + shifts.reshape((-1, 1, 4))
    return all_anchors.reshape((-1, 4))



def compute_shape(image_shape, pyramid_levels):
    """Compute shapes based on pyramid levels.

    :param image_shape:
    :param pyramid_levels:
    :return:
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


def anchors_for_shape(image_shape, pyramid_levels=None, ratios=None, scales=None,
                      strides=None, sizes=None, shapes_callback=None):
    """
    Compute anchors for a given input shape and pyramid levels.

    :param image_shape: Shape of the input image.
    :param pyramid_levels: List of pyramid levels.
    :param ratios: Array of anchor aspect ratios.
    :param scales: Array of anchor scales.
    :param strides: List of anchor strides corresponding to each pyramid level.
    :param sizes: List of anchor sizes for each pyramid level.
    :param shapes_callback: Callback function to compute shapes based on pyramid levels.
    :return: Array of computed anchors.
    """
    image_shapes = compute_shape(image_shape, pyramid_levels)
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors = generate_anchor_boxes(base_size=sizes[idx], ratios=ratios, scales=scales)
        shifted_anchors = shift(image_shapes[idx], strides[idx], anchors)
        all_anchors = np.append(all_anchors, shifted_anchors, axis=0)
    return all_anchors


def tests():
    image_shape = (800, 1200)
    pyramid_levels = [3, 4, 5, 6, 7]
    ratios = np.array([0.5, 1, 2])
    scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
    strides = [2 ** x for x in pyramid_levels]
    sizes = [2 ** (x + 2) for x in pyramid_levels]
    generated_anchors = anchors_for_shape(image_shape=(800, 1200), pyramid_levels=pyramid_levels,
                                          ratios=ratios, scales=scales, strides=strides, sizes=sizes)
    print("Shape of generated Anchors:")
    print(generated_anchors.shape)

    print("Shifted boxes", shift((5, 5), stride=2, anchors=np.array([[0, 0, 10, 10]])))
    print("Liczba anchor boxów: ", len(generate_anchor_boxes(16)))

    anchors = Anchors()
    input_tensor = torch.randn((1, 3, 224, 224))  # Przykładowy tensor wejściowy
    boxes = anchors(input_tensor)
    print(boxes.shape)


def tests_Anchor():
    anchors = Anchors()
    input_tensor = torch.randn((1, 3, 64, 64))  # Przykładowy tensor wejściowy
    boxes = anchors(input_tensor)
    print(boxes.shape)


if __name__ == "__main__":
    tests_Anchor()
