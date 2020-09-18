import torch
import albumentations as A
import os
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
import torch.optim as optim
import math
import json

from efficientnet_pytorch import EfficientNet
from glob import glob
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

parser = ArgumentParser()
parser.add_argument("--img_folder", type=str)
parser.add_argument("--mode", type=str, default="")

args = parser.parse_args()
if args.mode == "tta":
    h_flips = [0.0, 0.0, 1.0, 1.0]
    v_flips = [0.0, 1.0, 0.0, 1.0]
else:
    h_flips = [0.0]
    v_flips = [0.0]


paths = glob(os.path.join(args.img_folder, '**/*.jpg'), recursive=True)
model =  EfficientNet.from_pretrained('efficientnet-b3', weights_path='./best_checkp_gender_effnet-b3_focal.pth', num_classes=2)
model.load_state_dict(torch.load('./best_checkp_gender_effnet-b3_focal.pth'))
model.cuda()
model.eval()


json_data = {}
for i in tqdm(range(len(paths))):
    path = paths[i]
    img_name = path.rsplit('/', 1)[-1]
    img = np.array(Image.open(path))
    result_probs = []
    for h_flip, v_flip in zip(h_flips, v_flips):
        aug = A.Compose({
                A.Resize(224, 224),
                A.HorizontalFlip(p=h_flip),
                A.VerticalFlip(p=v_flip),
                A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                })
        aug_img = aug(image=img)['image']
        aug_img = np.transpose(img, (2, 0, 1)).astype(np.float32)

        aug_tensor = torch.tensor(aug_img).unsqueeze(0).cuda()

        with torch.no_grad():
            probs = F.softmax(model(aug_tensor), dim=-1)
            #print(probs)
            result_probs.append(probs)

    ans = torch.argmax(torch.sum(torch.stack(result_probs), dim=0)/len(h_flips))
    if ans.item() == 0:
        json_data[img_name] = 'female'
    else:
        json_data[img_name] = 'male'

with open('process_results.json', 'w') as f:
    json.dump(json_data, f)




