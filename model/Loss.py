import torch
import torch.nn as nn
import torch.nn.functional as F

class RetinaNetBoxLoss(nn.Module):
    def __init__(self, delta):
        """
        Inicjalizacja funkcji straty RetinaNetBoxLoss.

        Parametry:
        - delta: Parametr określający próg, poniżej którego straty stają się kwadratowe.
        """
        super(RetinaNetBoxLoss, self).__init__()
        self.delta = delta

    def forward(self, y_true, y_pred):
        """
        Metoda forward obliczająca stratę RetinaNetBoxLoss.

        Parametry:
        - y_true: Tensor z prawdziwymi boxami (ground truth), zawierającymi informacje o położeniu obiektów.
        - y_pred: Tensor z przewidywanymi wartościami, które model ma zwrócić (przewidywane położenie obiektów).

        Zwraca:
        - Tensor zawierający sumę strat dla każdego przykładu.
        """
        difference = y_true - y_pred
        absolute_difference = torch.abs(difference)
        squared_difference = difference ** 2
        loss = torch.where(absolute_difference < self.delta, 0.5 * squared_difference, absolute_difference - 0.5)
        return torch.sum(loss, dim=-1)


class RetinaFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        """
        Inicjalizacja funkcji straty Focal Loss.

        Parametry:
        - alpha: Parametr wpływający na wagę klas pozytywnych i negatywnych.
        - gamma: Parametr regulujący skupienie na trudnych do sklasyfikowania przykładach.
        """
        super(RetinaFocalLoss, self).__init__()
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
        # Obliczenia cross entropy z funkcją sigmoid
        cross_entropy = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')

        # Obliczenia prawdopodobieństw przy użyciu funkcji sigmoid
        probs = torch.sigmoid(y_pred)

        # Określenie wag dla klas pozytywnych i negatywnych
        alpha = torch.where(y_true == 1, self.alpha, 1 - self.alpha)

        # Obliczenie pt - czynnika skupiającego uwagę na trudnych przykładach
        pt = torch.where(y_true == 1, probs, 1 - probs)

        # Obliczenia straty za pomocą Focal Loss
        loss = alpha * (1 - pt)**self.gamma * cross_entropy

        # Sumowanie strat wzdłuż ostatniej osi tensora
        return torch.sum(loss, dim=-1)


class RetinaNetLoss(nn.Module):
    def __init__(self, num_classes=1, alpha=0.25, gamma=2.0, delta=1.0):
        """
        Inicjalizacja klasy RetinaNetLoss.

        Args:
            num_classes (int): Liczba klas w zadaniu detekcji obiektów.
            alpha (float): Parametr kontrolujący wagę klas pozytywnych i negatywnych w klasyfikacji.
            gamma (float): Parametr gamma używany w funkcji Focal Loss.
            delta (float): Parametr delta używany w funkcji RetinaNetBoxLoss.
        """
        super(RetinaNetLoss, self).__init__()
        self.clf_loss = RetinaFocalLoss(alpha=alpha, gamma=gamma)
        self.box_loss = RetinaNetBoxLoss(delta)
        self.num_classes = num_classes

    def forward(self, y_true, y_pred):
        """
        Metoda obliczająca łączną stratę dla modelu RetinaNet.

        Args:
            y_true (torch.Tensor): Tensor zawierający rzeczywiste etykiety.
            y_pred (torch.Tensor): Tensor zawierający przewidywane wartości.

        Returns:
            torch.Tensor: Obliczona łączna strata dla modelu RetinaNet.
        """
        y_pred = y_pred.float()
        box_labels = y_true[:, :, :4]
        box_predictions = y_pred[:, :, :4]
        temp = y_true.clone()
        temp[:, :, 4] = torch.where(temp[:, :, 4] == -1, torch.tensor(0), temp[:, :, 4])
        cls_labels = F.one_hot(
            temp[:, :, 4].long(),
            num_classes=self.num_classes + 1,
        ).float()
        cls_predictions = y_pred[:, :, 4:]

        # Stworzenie masek dla pozytywnych przykładów i ignorowania
        positive_mask = (y_true[:, :, 4] > -1.0).float()
        ignore_mask = (y_true[:, :, 4] == -2.0).float()

        # Obliczenie straty klasyfikacji i regresji bounding boxów
        clf_loss = self.clf_loss(cls_labels, cls_predictions)
        box_loss = self.box_loss(box_labels, box_predictions)

        # Zerowanie strat tam, gdzie maski są równe 1.0
        clf_loss = torch.where(ignore_mask.eq(1.0), torch.tensor(0.0), clf_loss)
        box_loss = torch.where(positive_mask.eq(1.0), box_loss, torch.tensor(0.0))

        # Obliczenie średnich strat, normalizowanych przez sumę pozytywnych przykładów
        normalizer = positive_mask.sum(dim=-1)
        clf_loss = torch.div(clf_loss.sum(dim=-1), normalizer)
        box_loss = torch.div(box_loss.sum(dim=-1), normalizer)

        # Ostateczna strata to suma straty klasyfikacji i regresji bounding boxów
        loss = clf_loss + box_loss
        return loss


def test_box_loss():
    y_true = torch.tensor([[1, 2, 4, 5], [3, 1, 6, 4]], dtype=torch.float32)
    y_pred = torch.tensor([[0, 1, 3, 4], [2, 2, 4, 5]], dtype=torch.float32)
    delta = 1.0
    retina_loss = RetinaNetBoxLoss(delta)
    loss = retina_loss(y_true, y_pred)
    print("\nGround Truth:")
    print(y_true)
    print("\nPredictions:")
    print(y_pred)
    print("\nLoss:")
    print(loss)


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
    test_box_loss()
    test_focal_loss()
    test_retina_loss()
