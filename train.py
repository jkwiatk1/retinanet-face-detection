import collections

import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, ToPILImage, Resize
import torch.optim as optim

from WiderDataLoader.wider_loader import WiderFaceDataset
from WiderDataLoader.wider_batch_iterator import BatchIterator
from model.RetinaNet import RetinaNet, evaluate

DATA_DIR = 'WIDER'
NUM_CLASSES = 1
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
EPOCHS_NUM = 10
WEIGHTS = 'data/model'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

transform = Compose(
    [ToPILImage(), Resize((800, 1024)), ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

wider_train_dataset = WiderFaceDataset(DATA_DIR, split='train', transform=transform)
wider_val_dataset = WiderFaceDataset(DATA_DIR, split='val', transform=transform)
# wider_test_dataset = WiderFaceDataset(DATA_DIR, split='test', transform=transform)

train_data = BatchIterator(wider_train_dataset, batch_size=BATCH_SIZE, shuffle=False)
# val_data = BatchIterator(wider_val_dataset, batch_size=BATCH_SIZE, shuffle=False)
# test_data = BatchIterator(wider_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = RetinaNet().to(device)
# for _, data in enumerate(train_data):
#     pred = model(data['img'].cuda().float())
#     print(pred)
#     print(type(data['img']))
#     size = data['img'].shape
#     print(size)
#     break

#print(model)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
model.train()

loss_hist = collections.deque(maxlen=500)

for epoch_num in range(EPOCHS_NUM):
    model.train()

    epoch_loss = []

    for iter_num, data in enumerate(train_data):
        optimizer.zero_grad()
        if torch.cuda.is_available():
            classification_loss, regression_loss = model([data['img'].cuda().float(), data['boxes_list'].cuda()])
        else:
            classification_loss, regression_loss = model([data['img'].float(), data['boxes_list']])

        loss = classification_loss + regression_loss

        if bool(loss == 0):
            continue

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        optimizer.step()

        loss_hist.append(float(loss))

        epoch_loss.append(float(loss))

        print(
            'Epoch: {}/{} | Iteration: {}/{} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                epoch_num, EPOCHS_NUM, iter_num, len(train_data), float(classification_loss), float(regression_loss), np.mean(loss_hist)))

        del classification_loss
        del regression_loss
    filename = f'model_{epoch_num}.pth'
    torch.save(model, WEIGHTS + filename)
    model.eval()
    evaluate(dataset=wider_val_dataset, model=model, threshold=0.05)


