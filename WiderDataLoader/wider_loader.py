import math
import os
import re

import numpy as np
import skimage
from PIL import Image
import torch
from matplotlib import pyplot as plt
from sympy import transpose
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, ToPILImage, Resize
import torchvision.transforms as transforms
from torchvision.ops import box_convert
from model.Utils import show_image


class WiderFaceDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, difficulty='low'):
        self.split = split
        self.data_dir = data_dir
        self.transform = transform
        self.difficulty = difficulty
        self.boxes = []
        self.boxes_num = []
        self.parameters = []
        datasets.WIDERFace(root=data_dir, split=split, transform=transform, download=True)
        self.image_paths = self._get_image_paths()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Handle slicing for a range of indices [start:stop]
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else len(self.image_paths)
            step = idx.step if idx.step is not None else 1
            selected_indices = range(start, stop, step)
        else:
            # Handle a single index
            selected_indices = [idx]

        samples = []
        for selected_idx in selected_indices:
            img_path = self.image_paths[selected_idx]
            box_num = self.boxes_num[selected_idx]
            boxes = self.boxes[selected_idx]
            param = self.parameters[selected_idx]
            image = self.load_image(img_path)
            begin_resolution = image.shape[:2]
            if self.transform:
                image = self.transform(image)
            end_resolution = image.shape[:2]
            if isinstance(image, torch.Tensor):
                end_resolution = image.shape[-2:]
            boxes_adjusted = self.adjust_boxes(boxes, begin_resolution, end_resolution)
            if self.split == 'train':
                image, boxes_adjusted = self.augment_image_and_boxes(image, boxes_adjusted)
            sample = {'img': image, 'boxes_num': box_num, 'boxes_list': boxes_adjusted, 'parameters': param}
            samples.append(sample)

        if len(samples) == 1:
            return samples[0]
        else:
            return samples

    def _get_image_paths(self):
        image_paths = []
        if self.split == 'train' or self.split == 'val':
            annotations_path = os.path.join(self.data_dir, 'widerface', 'wider_face_split',
                                            'wider_face_' + self.split + '_bbx_gt.txt')
            with open(annotations_path, 'r') as file:
                lines = file.readlines()
                for i in range(len(lines)):
                    if ".jpg" in lines[i]:
                        path = os.path.join(self.data_dir, 'widerface', 'WIDER_' + self.split, 'images', lines[i].strip())
                        match = re.search(r'\d+', path)
                        if match:
                            first_number = int(match.group())
                        if self.difficulty == 'normal' and first_number <= 20:
                            continue
                        if self.difficulty == 'low' and first_number <= 40:
                            continue
                        image_paths.append(path)
                        number = int(lines[i + 1])
                        self.boxes_num.append(number)
                        boxes = []
                        parameters = []
                        for j in range(number):
                            numbers = list(map(int, lines[i + 2 + j].split()))
                            coordinates = {"x": numbers[0], "y": numbers[1],
                                           "w": numbers[2], "h": numbers[3]}
                            param = {'blur': numbers[4], 'expression': numbers[5], 'illumination': numbers[6],
                                          'invalid': numbers[7], 'occlusion': numbers[8], 'pose': numbers[9]}
                            boxes.append(coordinates)
                            parameters.append(param)
                        self.boxes.append(boxes)
                        self.parameters.append(parameters)
        return image_paths

    def load_image(self, path):
        img = skimage.io.imread(path)
        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)
        return img

    def transform_boxes(self, boxes, angle, scale, image_size):
        """
        MSkalowanie boxów i obracanie

        Argumenty:
        - box:  ramka w formacie: [x, y, width, height, class]
        - angle: kąt w stopniach
        - angle: parametr skalowania boxa
        - image_size - rozmiar obrazu przed przekształceniem.
        """
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        h, w = image_size
        new_boxes = []
        for box in boxes:
            x_left, y_top, bw, bh, cl = box

            # Wyznaczenie narożników
            corners = [
                [x_left, y_top],  # Lewy górny
                [x_left + bw, y_top],  # Prawy górny
                [x_left, y_top + bh],  # Lewy dolny
                [x_left + bw, y_top + bh]  # Prawy dolny
            ]

            # Transformacja narożników
            transformed_corners = []
            for (x, y) in corners:
                # Obrót wokół środka obrazu
                new_x = cos_a * (x - w/2) - sin_a * (y - h/2) + w/2
                new_y = sin_a * (x - w/2) + cos_a * (y - h/2) + h/2

                # Skalowanie
                new_x = (new_x - w/2) * scale + w/2
                new_y = (new_y - h/2) * scale + h/2

                transformed_corners.append([new_x, new_y])

            # Obliczenie nowych współrzędnych boxa
            x_coords, y_coords = zip(*transformed_corners)
            new_x_left = min(x_coords)
            new_y_top = min(y_coords)
            new_bw = max(x_coords) - new_x_left
            new_bh = max(y_coords) - new_y_top

            new_boxes.append(torch.tensor([new_x_left, new_y_top, new_bw, new_bh, cl]))

        return torch.stack(new_boxes)

    def augment_image_and_boxes(self, image, boxes, angle_range=(-180, 180), scale_range=(-0.2, 0.2)):
        angle = torch.FloatTensor(1).uniform_(*angle_range).item()
        scale = torch.FloatTensor(1).uniform_(1 + scale_range[0], 1 + scale_range[1]).item()

        transform = transforms.Compose([
            transforms.RandomAffine([angle, angle], scale=(scale, scale))
        ])
        transformed_image = transform(image)

        if boxes.shape[1] > 0 and boxes[0, 0] != -1:
            transformed_boxes = self.transform_boxes(boxes, angle, scale, image_size=image.shape[-2:])
        else:
            transformed_boxes = boxes
        return transformed_image, transformed_boxes

    def adjust_boxes(self, original_boxes, original_resolution, target_resolution):
        """
        Adjust bounding boxes coordinates and sizes based on the original and target resolutions.
        Parameters:
            original_boxes (list): List of boxes [x, y, width, height].
            original_resolution (tuple): Tuple representing the original resolution (original_height, original_width).
            target_resolution (tuple): Tuple representing the target resolution (target_height, target_width).

        Returns:
            adjusted_boxes (list): List of adjusted boxes [x_adj, y_adj, width_adj, height_adj].
        """
        original_height, original_width = original_resolution
        target_height, target_width = target_resolution
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        adjusted_boxes = []
        for box in original_boxes:
            x, y, width, height = box['x'], box['y'], box['w'], box['h']
            x_adj = int(x * scale_x)
            width_adj = int(width * scale_x)
            y_adj = int(y * scale_y)
            height_adj = int(height * scale_y)
            adjusted_boxes.append([x_adj, y_adj, width_adj, height_adj, 0])
        adjusted_boxes = torch.tensor(adjusted_boxes)
        return adjusted_boxes


# Przykład użycia

'''
data_dir = '../WIDER'
transform = Compose([ToPILImage(), Resize((900, 1024)), ToTensor()])
wider_train_dataset = WiderFaceDataset(data_dir, 'train', transform)
data = wider_train_dataset[1]
show_image(data['img'], data['boxes_list'])

wider_val_dataset = WiderFaceDataset(data_dir, 'val', transform)
data2 = wider_val_dataset[0]
show_image(data2['img'], data2['boxes_list'])
'''


def histogram_of_paramets():
    data_dir = '../WIDER'
    transform = Compose([ToPILImage(), Resize((900, 1024)), ToTensor()])
    wider_train_dataset = WiderFaceDataset(data_dir, split='train', transform=transform)
    blur = []
    expression = []
    illumination = []
    invalid = []
    occlusion = []
    pose = []
    box_width = []
    box_height = []
    for data in wider_train_dataset:
        for param in data['parameters']:
            blur.append(param['blur'])
            expression.append(param['expression'])
            illumination.append(param['illumination'])
            invalid.append(param['invalid'])
            occlusion.append(param['occlusion'])
            pose.append(param['pose'])
        for box in data['boxes_list']:
            box_width.append(box[2])
            box_height.append(box[3])

    plt.figure(figsize=(15, 10))
    plt.subplot(4, 2, 1)
    plt.hist(blur, bins=[0, 1, 2, 3], align='left', edgecolor='black')
    plt.title('Blur')
    plt.subplot(4, 2, 2)
    plt.hist(expression, bins=[0, 1, 2], align='left', edgecolor='black')
    plt.title('Expression')
    plt.subplot(4, 2, 3)
    plt.hist(illumination, bins=[0, 1, 2], align='left', edgecolor='black')
    plt.title('Illumination')
    plt.subplot(4, 2, 4)
    plt.hist(invalid, bins=[0, 1, 2], align='left', edgecolor='black')
    plt.title('Invalid')
    plt.subplot(4, 2, 5)
    plt.hist(occlusion, bins=[0, 1, 2, 3], align='left', edgecolor='black')
    plt.title('Occlusion')
    plt.subplot(4, 2, 6)
    plt.hist(pose, bins=[0, 1, 2], align='left', edgecolor='black')
    plt.title('Pose')
    plt.subplot(4, 2, 7)
    plt.hist(box_width, bins='auto', align='left', edgecolor='black', density=True)
    plt.title('Box width')
    plt.subplot(4, 2, 8)
    plt.hist(box_height, bins='auto', align='left', edgecolor='black', density=True)
    plt.title('Box height')
    plt.tight_layout()
    plt.show()


def histogram_of_resolution():
    he = []
    wi = []
    data_dir = '../WIDER'
    transform = Compose([ToPILImage(), Resize((900, 1024)), ToTensor()])
    dataset_train = WiderFaceDataset(data_dir, split='train', transform=transform)
    for i in range(len(dataset_train)):
        h, w = dataset_train[i]['img'].shape[:2]
        if h > 2000:
            continue
        he.append(h)
        wi.append(w)
    print('Srednia szerokosc: ', sum(wi)/len(wi))
    print('Srednia wysokosc: ', sum(he) / len(he))
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.hist(wi, bins=20, color='blue', alpha=0.7)
    plt.title('Histogram - szerokosc')
    plt.subplot(2, 1, 2)
    plt.hist(he, bins=20, color='blue', alpha=0.7)
    plt.title('Histogram - wysokosc')
    plt.savefig("histogram_wider_height", bbox_inches='tight', pad_inches=0)
    plt.show()

# histogram()


