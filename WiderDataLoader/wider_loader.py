import os
import re

import numpy as np
import skimage
from PIL import Image
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, ToPILImage, Resize


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
wider_train_dataset = WiderFaceDataset(data_dir, 'train')
print(wider_train_dataset[0])
wider_val_dataset = WiderFaceDataset(data_dir, 'val')
print(wider_val_dataset[0])
'''

def histogram():
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


def histogram2():
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


