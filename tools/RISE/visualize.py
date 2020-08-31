import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import Image
import cv2
import glob

from RISE.utils import *
from RISE.explanations import RISE

cudnn.benchmark = True

# mapsPaths = glob.glob("/home/cse/staff/muku95.cstaff/scratch/BSTI_Attention/*")

def reverse(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    # Mean and std for ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def visualize(model,img_dir,visualize_dir,CovidDataLoader):
    
    dataset = CovidDataLoader(image_dir=img_dir, transform=preprocess)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
    gpu_batch_size = 20

    model = model.eval()
    model = model.cuda()

    for p in model.parameters():
        p.requires_grad = False
        
    explainer = RISE(model, (224,224), gpu_batch_size)
    
    # Generate masks for RISE or use the saved ones.
    maskspath = 'masks.npy'
    generate_new = True

    if generate_new or not os.path.isfile(maskspath):
        explainer.generate_masks(N=10000, s=8, p1=0.1, savepath=maskspath)
    else:
        explainer.load_masks(maskspath)
    
    def explain_all(data_loader, explainer):
        # Get all predicted labels first
        target = np.empty(len(data_loader), np.int)
        for i, (img, _) in enumerate(tqdm(data_loader, total=len(data_loader), desc='Predicting labels')):
            p, c = torch.max(model(torch.autograd.Variable(img.cuda()))[0], dim=1)
            target[i] = c[0]

        # Get saliency maps for all images in val loader
        explanations = np.empty((len(data_loader), 224,224))
        for i, (img, _) in enumerate(tqdm(data_loader, total=len(data_loader), desc='Explaining images')):
            saliency_maps = explainer(torch.autograd.Variable(img.cuda()))
            explanations[i] = saliency_maps[target[i]].cpu().numpy()
        return explanations

    explanations = explain_all(data_loader, explainer)
    cm = plt.get_cmap("jet")
    if not os.path.exists(visualize_dir):
        os.makedirs(visualize_dir)
    for i, (img, imgName) in enumerate(tqdm(data_loader, total=len(data_loader), desc='Generating visualizations')):
        img_name = os.path.splitext(imgName[0])[0]
        p, c = torch.max(model(torch.autograd.Variable(img.cuda()))[0], dim=1)
        p, c = p.data[0], c.data[0]
        
        ######
#         index=0
#         for index,fpath in enumerate(mapsPaths):
#             aName = mapsPaths[index].split("/")[-1][:-4]
#             if "vis_"+img_name==aName:
#                 break
#         imageAt = Image.open(mapsPaths[index]).convert('RGB')
#         imgPre = preprocess(imageAt)
        
        ######
        
        plt.figure(figsize=(16, 10))
        plt.subplot(121)
        plt.axis('off')
        tensor_imshow(img[0])
        plt.subplot(122)
        plt.axis('off')
        tensor_imshow(img[0])
        sal = explanations[i]
        
#         print(sal)
        
        ## Heat map for cv2 image
#         sal2 = (sal*256).astype(np.uint8)
#         heatmap =cv2.applyColorMap(sal2, cv2.COLORMAP_JET)
#         imgVal = reverse(img[0])
#         rescaled= np.uint8(imgVal*256)

#         fin = cv2.addWeighted(heatmap, 0.75, rescaled, 0.25, 0)
#         img_pil = Image.fromarray(fin).convert('RGB')
#         img_pil = img_pil.resize((480,480))
#         img_pil.save(visualize_dir+"/"+imgName[0])


        #########################
        plt.imshow(sal, cmap='jet', alpha=0.25)
        plt.savefig(visualize_dir+"/"+img_name+"_visualization.png")
