import random
from torch.nn.utils.rnn import pad_sequence
import torch
from WiderDataLoader.wider_loader import WiderFaceDataset


class BatchIterator:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.current_index = 0
        self.indices = list(range(len(dataset)))
        if self.shuffle:
            random.shuffle(self.indices)


    def __iter__(self):
        self.current_index = 0
        if self.shuffle:
            random.shuffle(self.indices)
        return self


    def __next__(self):
        if self.current_index >= len(self.dataset):
            # Wszystkie dane zostały już przetworzone
            raise StopIteration
        indices = self.indices[self.current_index:self.current_index + self.batch_size]
        data = [self.dataset[i] for i in indices]
        images =[]
        boxes_num = []
        boxes = []
        for i in range(len(data)):
            images.append(data[i]['img'])
            boxes_num.append((data[i]['boxes_num']))
            if data[i]['boxes_num'] != 0:
                boxes.append(data[i]['boxes_list'])
            else:
                boxes.append(torch.tensor([[-1, -1, -1, -1, -1]]))
        self.current_index += self.batch_size
        images = torch.stack(images, dim=0)
        padded_tensors = pad_sequence(boxes, batch_first=True, padding_value=-1)
        batch = {'img': images, 'boxes_num': boxes_num, 'boxes_list': padded_tensors}
        return batch

    def __len__(self):
        return len(self.dataset) // self.batch_size
