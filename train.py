import collections

import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, ToPILImage, Resize
import torch.optim as optim

from WiderDataLoader.wider_loader import WiderFaceDataset
from WiderDataLoader.wider_batch_iterator import BatchIterator
from model.RetinaNet import RetinaNet, evaluate
from model.Utils import show_image

DATA_DIR = 'WIDER'
NUM_CLASSES = 1
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
EPOCHS_NUM = 100
SCHEDULER_GAMMA = 0.7
WEIGHTS = 'data/model/'
PRETRAIN_WEIGHTS = 'data/model/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

transform = Compose(
    [ToPILImage(), Resize((156, 156)), ToTensor(),Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

wider_train_dataset = WiderFaceDataset(DATA_DIR, split='train', transform=transform, augment=True)
wider_val_dataset = WiderFaceDataset(DATA_DIR, split='val', transform=transform, augment=False)
# wider_test_dataset = WiderFaceDataset(DATA_DIR, split='test', transform=transform)

train_data = BatchIterator(wider_train_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_data = BatchIterator(wider_val_dataset, batch_size=BATCH_SIZE, shuffle=False)
# test_data = BatchIterator(wider_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = RetinaNet().to(device)
# model = torch.load(PRETRAIN_WEIGHTS+'one_batch_train_lr_2e4_from_pretrain/model_120.pth', map_location=device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=EPOCHS_NUM/10, gamma=SCHEDULER_GAMMA)


loss_hist = collections.deque(maxlen=500)
regr_hist = collections.deque(maxlen=500)
clas_hist = collections.deque(maxlen=500)
loss_val_hist = collections.deque(maxlen=500)
regr_val_hist = collections.deque(maxlen=500)
clas_val_hist = collections.deque(maxlen=500)
best_valid_loss = float('Inf')

for epoch_num in range(EPOCHS_NUM):
    epoch_loss = []
    regr_loss = []
    clas_loss = []
    model.train()
    for iter_num, data in enumerate(train_data):
        if iter_num >= 1:
            break
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
        print(
            f'Krok: {iter_num}/{len(train_data)}')
    average_loss = np.mean(epoch_loss)
    average_clas_loss = np.mean(clas_loss)
    average_regr_loss = np.mean(regr_loss)
    loss_hist.append(average_loss)
    regr_hist.append(average_regr_loss)
    clas_hist.append(average_clas_loss)
    print(
        f'Epoch: {epoch_num}/{EPOCHS_NUM} | Average Loss: {average_loss} | Classification loss: {float(average_clas_loss):1.5f} | Regression loss: {float(average_regr_loss):1.5f}')
    scheduler.step()

    epoch_loss = []
    regr_loss = []
    clas_loss = []
    for iter_num, data in enumerate(val_data):
        if iter_num >= 1:
            break
        if torch.cuda.is_available():
            classification_loss, regression_loss = model([data['img'].cuda().float(), data['boxes_list'].cuda()])
        else:
            classification_loss, regression_loss = model([data['img'].float(), data['boxes_list']])
        loss = classification_loss + regression_loss
        if bool(loss == 0):
            continue
    average_loss = np.mean(epoch_loss)
    average_clas_loss = np.mean(clas_loss)
    average_regr_loss = np.mean(regr_loss)
    loss_val_hist.append(average_loss)
    regr_val_hist.append(average_regr_loss)
    clas_val_hist.append(average_clas_loss)
    print(
        f'Evaluate | Average Loss: {average_loss} | Classification loss: {float(average_clas_loss):1.5f} | Regression loss: {float(average_regr_loss):1.5f}')

    filename = f'model_{epoch_num}.pth'
    #torch.save(model, WEIGHTS + filename)
    model.eval()
    try:
        evaluate(dataset=wider_train_dataset[0:BATCH_SIZE], model=model, threshold=0.05)
    except Exception as e:
        print(f"Error: {e}")
