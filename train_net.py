import torch
import albumentations as A
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
import torch.optim as optim
import math
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet
from torch.autograd import Variable
from glob import glob
from PIL import Image
from argparse import ArgumentParser
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

#Dataset class for gender classification
class FaceDataset(Dataset):

    def __init__(self, root, folder_names, transforms=None):

        self.root = root
        self.folder_names = folder_names
        self.transforms = transforms

        #Finding all files in this folders with *.jpg extension for male and female separately
        self.males = glob(os.path.join(root, folder_names[0], '*.jpg'))
        self.females = glob(os.path.join(root, folder_names[1], '*.jpg'))

        self.aug = transforms

    def __len__(self):
        return len(self.males) + len(self.females)

    def __getitem__(self, idx):
        if idx >= len(self.males):
            img_path = self.females[idx % len(self.males)]
            label = torch.tensor([0])
        else:
            img_path = self.males[idx]
            label = torch.tensor([1])
        
        img = np.array(Image.open(img_path))
        
        if self.transforms:
            img = self.aug(image=img)['image']
            img = np.transpose(img, (2, 0, 1)).astype(np.float32)

        img = torch.tensor(img)
        return {"img": img, "label": label}

parser = ArgumentParser()
parser.add_argument("--root_dir", type=str)
parser.add_argument("--male_dir_name", type=str)
parser.add_argument("--female_dir_name", type=str)

args = parser.parse_args()

#Creating augmentations for training, which will help generalization of the model
aug =  A.Compose({
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=(-90, 90)),
        A.VerticalFlip(p=0.5),
        A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        })

#creation of common dataset
dataset = FaceDataset(args.root_dir, [args.male_dir_name, args.female_dir_name], transforms=aug)

#creation of train and val datasets from common with *val_split* ratio
val_split = 0.2
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset) - math.floor(len(dataset) * val_split),
                                                                     math.floor(len(dataset) * val_split)])
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=32)


#Function which takes model (nn.Module) and validation dataloader to iterate through it and measure accuracy
def validate(model, val_loader):

    model.eval()
    correct = 0
    for idx, data in enumerate(val_loader):

        images, labels = data['img'], data['label']
        images = images.cuda()
        labels = labels.cuda()
        labels = labels.squeeze()

        predicts = torch.argmax(model(images), dim=1)
        correct += torch.sum((predicts == labels)).item()

    return correct / (len(val_loader)*val_loader.batch_size)


epochs = 15
#Using efficientnet-b3 with focal loss
model = EfficientNet.from_pretrained('efficientnet-b3',weights_path='./efficientnet-b3-5fb5a3c3.pth',
                                     num_classes=2).cuda()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)
criterion = FocalLoss()

best_acc = 0.0
for epoch in range(epochs):

    model.train()
    running_loss = 0.0
    for idx, data in enumerate(train_dataloader):
        
        optimizer.zero_grad()

        images, labels = data['img'], data['label']

        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if idx % 20 == 19:
            print("Epoch №{}, step №{}, curr_loss: {}".format(epoch + 1, idx + 1, running_loss / 20))
            running_loss = 0.0
    
    acc = validate(model, val_dataloader)
    print("Epoch №{}, val_acc: {}".format(epoch + 1, acc))
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), './best_checkp_gender_effnet.pth')


