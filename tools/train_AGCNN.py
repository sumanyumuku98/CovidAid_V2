import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data_v3 import ChestXrayDataSetTest, ChestXrayDataSet, CovidDataLoader
from PIL import Image, ImageDraw, ImageFont
from pprint import pprint


# from covidaid_v2 import CovidAID, Fusion_Branch
from model_AGCNN import CovidAidAttend, Fusion_Branch
import argparse
from tqdm import tqdm
import cv2
from torch.optim import lr_scheduler
import json
import datetime
# from sklearn.metrics import 
import time
from skimage.measure import label
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc, f1_score, classification_report
import seaborn as sn
import matplotlib
import matplotlib.pyplot as plt
from scipy import interp
from itertools import cycle
from collections.abc import Sequence, Iterable
import pandas as pd
import math
import random
import numbers
from skimage.draw import rectangle_perimeter, set_color
import types

import version_support_functional as F
try:
    import accimage
except ImportError:
    accimage = None



USE_GPU = torch.cuda.is_available()

if USE_GPU:
    print("Using GPU..")

# N_CLASSES = 3
# CLASSES = ["Normal", "Pneumonia", "Covid"]

TRAIN_LIST = './data/train_NEW2.txt'
VAL_LIST = './data/val_NEW2.txt'
TEST_LIST= './data/test_NEW2.txt'


save_model_name='AGCNN'

LR_G = 1e-3
LR_L = 1e-3
LR_F = 1e-4

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

def pred_converter(pred):
    new_pred=[]
    for i,arr in enumerate(pred):
        nonCovid = np.max(arr[:-1])
        covid = arr[-1]
        temp = np.array([nonCovid,covid])
        new_pred.append(temp)
    return np.stack(new_pred)



def labelConverter(y_pred):
    new_pred = []
    for i,label in enumerate(y_pred):
        if label==2:
            new_pred.append(1)
        else:
            new_pred.append(0)
    
    new_predNP = np.array(new_pred,dtype=np.uint)
    return new_predNP

class RandomAffine(object):
    """Random affine transformation of the image keeping center invariant

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
            will be apllied. Else if shear is a tuple or list of 2 values a shear parallel to the x axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a tuple or list of 4 values,
            a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (tuple or int): Optional fill color (Tuple for RGB Image And int for grayscale) for the area
            outside the transform in the output image.(Pillow>=5.0.0)

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and \
                    (len(shear) == 2 or len(shear) == 4), \
                    "shear should be a list or tuple and it must be of length 2 or 4."
                # X-Axis shear with [min, max]
                if len(shear) == 2:
                    self.shear = [shear[0], shear[1], 0., 0.]
                elif len(shear) == 4:
                    self.shear = [s for s in shear]
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            if len(shears) == 2:
                shear = [random.uniform(shears[0], shears[1]), 0.]
            elif len(shears) == 4:
                shear = [random.uniform(shears[0], shears[1]),
                         random.uniform(shears[2], shears[3])]
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, img):
        """
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
        return F.affine(img, *ret, resample=self.resample, fillcolor=self.fillcolor)

    def __repr__(self):
        s = '{name}(degrees={degrees}'
        if self.translate is not None:
            s += ', translate={translate}'
        if self.scale is not None:
            s += ', scale={scale}'
        if self.shear is not None:
            s += ', shear={shear}'
        if self.resample > 0:
            s += ', resample={resample}'
        if self.fillcolor != 0:
            s += ', fillcolor={fillcolor}'
        s += ')'
        d = dict(self.__dict__)
        d['resample'] = _pil_interpolation_to_str[d['resample']]
        return s.format(name=self.__class__.__name__, **d)




def Attention_gen_patchs(ori_image, fm_cuda,mode='normal'):
    # fm => mask =>(+ ori-img) => crop = patchs
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    minMaxList=[]
#     heatMapList=[]
    
    if USE_GPU:
        feature_conv = fm_cuda.data.cpu().numpy()
    else:
        feature_conv = fm_cuda.data.numpy()

    size_upsample = (224, 224)
    bz, nc, h, w = feature_conv.shape

    if USE_GPU:
        patchs_cuda = torch.FloatTensor().cuda()
    else:
        patchs_cuda = torch.FloatTensor()

    for i in range(0, bz):
        feature = feature_conv[i]
        cam = feature.reshape((nc, h*w))
        cam = cam.sum(axis=0)
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        
#         upsampled_img = cv2.resize(cam_img, size_upsample)
        heatmap_bin = binImage(cv2.resize(cam_img, size_upsample))
        heatmap_maxconn = selectMaxConnect(heatmap_bin)
        heatmap_mask = heatmap_bin * heatmap_maxconn
        
        ############################
        
#         heatMapList.append(heatmap_mask)
        
        
        ind = np.argwhere(heatmap_mask != 0)
#         print(len(ind[:,1]))
        
        if mode=='visualize':
#             print("Extracting Patch Dimensions for visualization")
            minh = min(np.sort(ind[:, 0])[3000:])
            minw = min(np.sort(ind[:, 1])[3000:])
            maxh = max(np.sort(ind[:, 0])[:-3000])
            maxw = max(np.sort(ind[:, 1])[:-3000])
        else:
            minh = min(ind[:, 0])
            minw = min(ind[:, 1])
            maxh = max(ind[:, 0])
            maxw = max(ind[:, 1])

        
        # to ori image
        image = ori_image[i].numpy().reshape(224, 224, 3)
        image = image[int(224*0.334):int(224*0.667),
                      int(224*0.334):int(224*0.667), :]
        
#         plt.imshow(image)

        image = cv2.resize(image, size_upsample)
#         plt.imshow(image)
        image_crop =(image[minh:maxh, minw:maxw, :]*std+mean) * 256
        
#         print(image_crop)
        
        # because image was normalized before
        
#         plt.imshow(image_crop)
        
        
        image_crop = preprocess(Image.fromarray(
            image_crop.astype('uint8')).convert('RGB'))
        
        
#         tensor_imshow(image_crop)
        
        minMaxList.append((minh,minw,maxh,maxw))
        
        if USE_GPU:
            img_variable = image_crop.view(3, 224, 224).unsqueeze(0).cuda()
        else:
            img_variable =image_crop.view(3, 224, 224).unsqueeze(0)
        
#         print(img_variable.shape)
        patchs_cuda = torch.cat((patchs_cuda, img_variable),0)

    return torch.autograd.Variable(patchs_cuda), minMaxList



def binImage(heatmap):
    _, heatmap_bin = cv2.threshold(
        heatmap, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # t in the paper
    #_, heatmap_bin = cv2.threshold(heatmap , 178 , 255 , cv2.THRESH_BINARY)
    return heatmap_bin


def selectMaxConnect(heatmap):
    labeled_img, num = label(heatmap, connectivity=2,
                             background=0, return_num=True)
    max_label = 0
    max_num = 0
    for i in range(1, num+1):
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)
    if max_num == 0:
        lcc = (labeled_img == -1)
    lcc = lcc + 0
    return lcc


def train(CKPT_PATH_INIT='init',CKPT_PATH_G=None, CKPT_PATH_L=None, CKPT_PATH_F=None,
          epochs=100, batch_size=32, logging=True, 
          save_dir=None,combine_pneumonia=True, freeze=False):
    print('********************load data********************')
    if combine_pneumonia:
        N_CLASSES = 3
        CLASSES = ["Normal", "Pneumonia", "Covid"]
    else:
        N_CLASSES= 4
        CLASSES=["Normal","Bacterial","Viral","Covid"]
    
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    train_dataset = ChestXrayDataSet(image_list_file=TRAIN_LIST,
                                     transform=transforms.Compose([
                                         transforms.Resize(256),
                                         RandomAffine(30, scale=(0.8,1.2), shear=[-15,15,-15,15]),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize,
                                     ]),
                                     combine_pneumonia=combine_pneumonia)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=8, pin_memory=True)

    val_dataset = ChestXrayDataSet(image_list_file=VAL_LIST,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize,
                                    ]),
                                    combine_pneumonia=combine_pneumonia)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=8, pin_memory=True)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if USE_GPU:
        Global_Branch_model = CovidAidAttend(combine_pneumonia).cuda()
        Local_Branch_model = CovidAidAttend(combine_pneumonia).cuda()
        Fusion_Branch_model = Fusion_Branch(input_size=2048, output_size=N_CLASSES).cuda()
    else:
        Global_Branch_model = CovidAidAttend(combine_pneumonia)
        Local_Branch_model = CovidAidAttend(combine_pneumonia)
        Fusion_Branch_model = Fusion_Branch(input_size=2048, output_size=N_CLASSES)
    
#     print("Existence of CKPT_INIT:{}".format(os.path.exists(CKPT_PATH_INIT)))
    
    if os.path.exists(CKPT_PATH_INIT):
        print("=> loading Initial checkpoint")
        if USE_GPU:
            checkpoint = torch.load(CKPT_PATH_INIT)
        else:
            checkpoint = torch.load(CKPT_PATH_INIT,map_location='cpu')
        Global_Branch_model.load_state_dict(checkpoint)
        Local_Branch_model.load_state_dict(checkpoint)
        print("=> Model weights initialized from CheXNet")

    else:
        print("=> Model training will be resumed")

        if os.path.exists(CKPT_PATH_G):
            if USE_GPU:
                checkpoint = torch.load(CKPT_PATH_G)
            else:
                checkpoint = torch.load(CKPT_PATH_G,map_location='cpu')
            Global_Branch_model.load_state_dict(checkpoint)
            print("=> loaded Global_Branch_model checkpoint")

        if os.path.exists(CKPT_PATH_L):
            if USE_GPU:
                checkpoint = torch.load(CKPT_PATH_L)
            else:
                checkpoint = torch.load(CKPT_PATH_L,map_location='cpu')

            Local_Branch_model.load_state_dict(checkpoint)
            print("=> loaded Local_Branch_model checkpoint")

        if os.path.exists(CKPT_PATH_F):
            if USE_GPU:
                checkpoint = torch.load(CKPT_PATH_F)
            else:
                checkpoint = torch.load(CKPT_PATH_F,map_location='cpu')

            Fusion_Branch_model.load_state_dict(checkpoint)
            print("=> loaded Fusion_Branch_model checkpoint")

    cudnn.benchmark = True
    criterion = nn.BCELoss()
    
    if freeze:
        print ("Freezing feature layers")
        for param in Global_Branch_model.densenet121.features.parameters():
            param.requires_grad = False
        for param in Local_Branch_model.densenet121.features.parameters():
            param.requires_grad = False
        
    
    optimizer_global = optim.Adam(Global_Branch_model.parameters(
    ), lr=LR_G, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    

    optimizer_local = optim.Adam(Local_Branch_model.parameters(
    ), lr=LR_L, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    


    optimizer_fusion = optim.Adam(Fusion_Branch_model.parameters(
    ), lr=LR_F, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    

    print('********************load model succeed!********************')
    

    print('********************begin training!********************')
    for epoch in range(epochs):
#         Global_Branch_model.train()
#         Local_Branch_model.train()
#         Fusion_Branch_model.train()
        since = time.time()
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)
        # set the mode of model
#         lr_scheduler_global.step()  # about lr and gamma
#         lr_scheduler_local.step()
#         lr_scheduler_fusion.step()
        Global_Branch_model.train()  # set model to training mode
        Local_Branch_model.train()
        Fusion_Branch_model.train()

        running_loss = 0.0
        # Iterate over data
        for i, (input_, target) in tqdm(enumerate(train_loader),total=len(train_loader)):
            
            if USE_GPU:
                input_var = torch.autograd.Variable(input_.cuda())
                target_var = torch.autograd.Variable(target.cuda())
            else:
                input_var = torch.autograd.Variable(input_)
                target_var = torch.autograd.Variable(target)
            
            optimizer_global.zero_grad()
            optimizer_local.zero_grad()
            optimizer_fusion.zero_grad()

            # compute output
            output_global, fm_global, pool_global = Global_Branch_model(
                input_var)

            patchs_var,_ = Attention_gen_patchs(input_, fm_global)

            output_local, _, pool_local = Local_Branch_model(patchs_var)
            # print(fusion_var.shape)
            output_fusion = Fusion_Branch_model(pool_global.data, pool_local.data)
            #

            # loss
            loss1 = criterion(output_global, target_var)
            loss2 = criterion(output_local, target_var)
            loss3 = criterion(output_fusion, target_var)
            #

            loss = loss1*0.4 + loss2*0.4 + loss3*0.2

            if (i % 100) == 0:
                print('step: %5d total_loss: %.3f loss_1: %.3f loss_2: %.3f loss_3: %.3f' %(
                      i, loss.data.cpu().numpy(), loss1.data.cpu().numpy(), loss2.data.cpu().numpy(),                                                 loss3.data.cpu().numpy()))

            loss.backward()
            optimizer_global.step()
            optimizer_local.step()
            optimizer_fusion.step()

            # print(loss.data.item())
            running_loss += loss.data.cpu().numpy()
            del input_var, target_var
            # break
            '''
            if i == 40:
                print('break')
                break
            '''
        
        if USE_GPU:
            torch.cuda.empty_cache()
        
        epoch_loss = float(running_loss) / float(i)
        print(' Epoch over  Loss: {:.5f}'.format(epoch_loss))

        print('*******testing!*********')
        val_metrics=test(Global_Branch_model, Local_Branch_model,
                         Fusion_Branch_model, val_loader,len(val_dataset),'val')
        
        timestamp = str(datetime.datetime.now()).split('.')[0]
        log= json.dumps({
            'timestamp':timestamp,
            'epoch': epoch+1,
            'train_loss': float('%.5f' % epoch_loss),
            'val_acc_G': float('%.5f' % val_metrics['valG']),
            'val_acc_L': float('%.5f' % val_metrics['valL']),
            'val_acc_F': float('%.5f' % val_metrics['valF'])
        })
        
        if logging:
            print(log)
            
        logFile= os.path.join(save_dir,'train.log')
        if logFile is not None:
            with open(logFile,'a') as f:
                f.write("{}\n".format(log))

        # break

        # save
        if epoch % 1 == 0:
            save_path = save_dir
            torch.save(Global_Branch_model.state_dict(), os.path.join(save_path,
                       save_model_name+'_Global'+'_epoch_'+str(epoch)+'.pth'))
            print('Global_Branch_model already save!')
            torch.save(Local_Branch_model.state_dict(), os.path.join(save_path,
                       save_model_name+'_Local'+'_epoch_'+str(epoch)+'.pth'))
            print('Local_Branch_model already save!')
            torch.save(Fusion_Branch_model.state_dict(), os.path.join(save_path,
                       save_model_name+'_Fusion'+'_epoch_'+str(epoch)+'.pth'))
            print('Fusion_Branch_model already save!')

        time_elapsed = time.time() - since
        print('Training one epoch complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        
def test(model_global, model_local, model_fusion, test_loader,val_size,mode='val',
         cm_path='cm',roc_path='roc',combine_pneumonia=True, binary_eval=False):
    
    if combine_pneumonia:
        N_CLASSES = 3
        CLASSES = ["Normal", "Pneumonia", "Covid"]
    else:
        N_CLASSES= 4
        CLASSES=["Normal","Bacterial","Viral","Covid"]

    # switch to evaluate mode
    BIN_Classes=["Non-Covid","Covid"]
    if mode=='val':
        
        model_global.eval()
        model_local.eval()
        model_fusion.eval()
        cudnn.benchmark = True
        global_correct=0.0
        local_correct=0.0
        fusion_correct=0.0

        for i, (inp, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
            if i % 2000 == 0:
                print('testing process:', i)
#         if USE_GPU:
#             target = target.cuda()
            

            gt = target
            if USE_GPU:
                input_var = torch.autograd.Variable(inp.cuda())
            else:
                input_var = torch.autograd.Variable(inp)

            #output = model_global(input_var)

            output_global, fm_global, pool_global = model_global(input_var)

            patchs_var,_ = Attention_gen_patchs(inp, fm_global)

            output_local, _, pool_local = model_local(patchs_var)

            output_fusion = model_fusion(pool_global.data, pool_local.data)

            pred_global = output_global.data.cpu()
            pred_local = output_local.data.cpu()
            pred_fusion = output_fusion.data.cpu()
        
            preds_global_labels= torch.max(pred_global,dim=-1)[1].numpy()
            preds_local_labels= torch.max(pred_local,dim=-1)[1].numpy()
            preds_fusion_labels=torch.max(pred_fusion,dim=-1)[1].numpy()
            gt_labels=torch.max(gt,dim=-1)[1].numpy()
        
            global_correct+=float(np.sum(preds_global_labels==gt_labels))
            local_correct+=float(np.sum(preds_local_labels==gt_labels))
            fusion_correct+=float(np.sum(preds_fusion_labels==gt_labels))

            del input_var
    

        if USE_GPU:
            torch.cuda.empty_cache()
    
        global_acc=global_correct/val_size
    
        local_acc=local_correct/val_size
    
        fusion_acc=fusion_correct/val_size
    
        val_metrics = {'valG':global_acc, 'valL':local_acc, 'valF': fusion_acc}
    
        return val_metrics
    
    else:
        gt = torch.FloatTensor()
        pred_global = torch.FloatTensor()
        pred_local = torch.FloatTensor()
        pred_fusion = torch.FloatTensor()
        
        model_global.eval()
        model_local.eval()
        model_fusion.eval()
        cudnn.benchmark = True
        
        image_name_list=[]
        
        for i, (inp, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
            if i % 2000 == 0:
                print('testing process:', i)
#         if USE_GPU:
#             target = target.cuda()
            
#             image_name = list(image_name)
#             image_name_list+=image_name
            gt = torch.cat((gt,target),0)
            if USE_GPU:
                input_var = torch.autograd.Variable(inp.cuda())
            else:
                input_var = torch.autograd.Variable(inp)

            #output = model_global(input_var)

            output_global, fm_global, pool_global = model_global(input_var)

            patchs_var,_ = Attention_gen_patchs(inp, fm_global)

            output_local, _, pool_local = model_local(patchs_var)

            output_fusion = model_fusion(pool_global.data, pool_local.data)
            
            pred_global = torch.cat((pred_global,output_global.data.cpu()),0)
            pred_local = torch.cat((pred_local,output_local.data.cpu()),0)
            pred_fusion = torch.cat((pred_fusion,output_fusion.data.cpu()),0)
            
            del input_var, output_global, fm_global, pool_global, patchs_var, output_local, pool_local

        if USE_GPU:
            torch.cuda.empty_cache()
        
#         print("Image Name Length:",len(image_name_list))
        print("GT Shape:", gt.shape)
        print("Global Shape:" ,pred_global.shape)
        print("Local: " ,pred_local.shape)
        print("Fusion:" ,pred_fusion.shape)
        
        if binary_eval:
            pred_global = torch.from_numpy(pred_converter(pred_global.numpy()))
            pred_local = torch.from_numpy(pred_converter(pred_local.numpy()))
            pred_fusion = torch.from_numpy(pred_converter(pred_fusion.numpy()))
            gt = torch.from_numpy(pred_converter(gt.numpy()))

        if binary_eval:
            aucGlobal = compute_AUC_scores(gt.numpy(),pred_global.numpy(),BIN_Classes)
            aucLocal = compute_AUC_scores(gt.numpy(),pred_local.numpy(),BIN_Classes)
            aucFusion= compute_AUC_scores(gt.numpy(),pred_fusion.numpy(),BIN_Classes)
        else:
            aucGlobal = compute_AUC_scores(gt.numpy(),pred_global.numpy(),CLASSES)
            aucLocal = compute_AUC_scores(gt.numpy(),pred_local.numpy(),CLASSES)
            aucFusion= compute_AUC_scores(gt.numpy(),pred_fusion.numpy(),CLASSES)
            
        
        CM_GLOBAL='%s_global'%cm_path
        CM_LOCAL='%s_local'%cm_path
        CM_FUSION='%s_fusion'%cm_path
        
        METRICS_GLOBAL='%s_metrics.txt'%CM_GLOBAL
        METRICS_LOCAL='%s_metrics.txt'%CM_LOCAL
        METRICS_FUSION='%s_metrics.txt'%CM_FUSION

        with open(METRICS_GLOBAL,'w') as file:
            file.write('The average AUROC is {auc_avg:.4f} \n'.format(auc_avg=aucGlobal[0]))
            for data in aucGlobal[1:]:
                file.write('The AUROC of {0:} is {1:.4f} \n'.format(data[0], data[1]))
                
        with open(METRICS_LOCAL,'w') as file:
            file.write('The average AUROC is {auc_avg:.4f} \n'.format(auc_avg=aucLocal[0]))
            for data in aucLocal[1:]:
                file.write('The AUROC of {0:} is {1:.4f} \n'.format(data[0], data[1]))
                
        with open(METRICS_FUSION,'w') as file:
            file.write('The average AUROC is {auc_avg:.4f} \n'.format(auc_avg=aucFusion[0]))
            for data in aucFusion[1:]:
                file.write('The AUROC of {0:} is {1:.4f} \n'.format(data[0], data[1]))
                
        ROC_GLOBAL='%s_global'%roc_path
        ROC_LOCAL='%s_local'%roc_path
        ROC_FUSION='%s_fusion'%roc_path
        
        if binary_eval:
            plot_ROC_curve(gt.numpy(),pred_global.numpy(),BIN_Classes,ROC_GLOBAL)
            plot_ROC_curve(gt.numpy(),pred_local.numpy(),BIN_Classes,ROC_LOCAL)
            plot_ROC_curve(gt.numpy(),pred_fusion.numpy(),BIN_Classes,ROC_FUSION)
        else:
            plot_ROC_curve(gt.numpy(),pred_global.numpy(),CLASSES,ROC_GLOBAL)
            plot_ROC_curve(gt.numpy(),pred_local.numpy(),CLASSES,ROC_LOCAL)
            plot_ROC_curve(gt.numpy(),pred_fusion.numpy(),CLASSES,ROC_FUSION)
            
        
        
        preds_global_labels=torch.max(pred_global,dim=-1)[1].numpy()
        preds_local_labels=torch.max(pred_local,dim=-1)[1].numpy()
        preds_fusion_labels=torch.max(pred_fusion,dim=-1)[1].numpy()
        gt_labels=torch.max(gt,dim=-1)[1].numpy()
        
#         preds_new_global = labelConverter(preds_global_labels)
#         preds_new_local = labelConverter(preds_local_labels)
#         preds_new_fusion = labelConverter(preds_fusion_labels)
#         gt_new = labelConverter(gt_labels)
        
        if binary_eval:
            plot_confusion_matrix(gt_labels, preds_global_labels,BIN_Classes,CM_GLOBAL)
            plot_confusion_matrix(gt_labels, preds_local_labels,BIN_Classes,CM_LOCAL)
            plot_confusion_matrix(gt_labels, preds_fusion_labels,BIN_Classes,CM_FUSION)
        else:
            plot_confusion_matrix(gt_labels, preds_global_labels,CLASSES,CM_GLOBAL)
            plot_confusion_matrix(gt_labels, preds_local_labels,CLASSES,CM_LOCAL)
            plot_confusion_matrix(gt_labels, preds_fusion_labels,CLASSES,CM_FUSION)


        

def compute_AUC_scores(y_true, y_pred, labels):
        """
        Computes the Area Under the Curve (AUC) from prediction scores

        y_true.shape  = [n_samples, n_classes]
        y_preds.shape = [n_samples, n_classes]
        labels.shape  = [n_classes]
        """
        aucRocMetricsList=[]
        AUROC_avg = roc_auc_score(y_true, y_pred)
        aucRocMetricsList.append(AUROC_avg)
        print('The average AUROC is {AUROC_avg:.4f}'.format(AUROC_avg=AUROC_avg))
        for y, pred, label in zip(y_true.transpose(), y_pred.transpose(), labels):
            print('The AUROC of {0:} is {1:.4f}'.format(label, roc_auc_score(y, pred)))
            aucRocMetricsList.append((label,roc_auc_score(y,pred)))
            
        return aucRocMetricsList


def plot_ROC_curve(y_true, y_pred, labels, roc_path): 
        """
        Plots the ROC curve from prediction scores

        y_true.shape  = [n_samples, n_classes]
        y_preds.shape = [n_samples, n_classes]
        labels.shape  = [n_classes]
        """
        n_classes = len(labels)
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for y, pred, label in zip(y_true.transpose(), y_pred.transpose(), labels):
            fpr[label], tpr[label], _ = roc_curve(y, pred)
            roc_auc[label] = auc(fpr[label], tpr[label])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[label] for label in labels]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for label in labels:
            mean_tpr += interp(all_fpr, fpr[label], tpr[label])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        lw = 2
        plt.plot(fpr["micro"], tpr["micro"],
                label='micro-average ROC curve (area = {0:0.3f})'
                    ''.format(roc_auc["micro"]),
                color='deeppink', linestyle=':', linewidth=2)

        plt.plot(fpr["macro"], tpr["macro"],
                label='macro-average ROC curve (area = {0:0.3f})'
                    ''.format(roc_auc["macro"]),
                color='navy', linestyle=':', linewidth=2)

        if len(labels) == 4:
            colors = ['green', 'cornflowerblue', 'darkorange', 'darkred']
        else:
            colors = ['green', 'cornflowerblue', 'darkred']
        for label, color in zip(labels, cycle(colors)):
            plt.plot(fpr[label], tpr[label], color=color, lw=lw,
                    label='ROC curve of {0} (area = {1:0.3f})'
                    ''.format(label, roc_auc[label]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        matplotlib.rcParams.update({'font.size': 14})
        plt.savefig('%s.png' % roc_path, pad_inches = 0, bbox_inches='tight')
        
        
def plot_confusion_matrix(y_true, y_pred, labels, cm_path):
        norm_cm = confusion_matrix(y_true, y_pred, normalize='true')
        norm_df_cm = pd.DataFrame(norm_cm, index=labels, columns=labels)
        plt.figure(figsize = (10,7))
        sn.heatmap(norm_df_cm, annot=True, fmt='.2f', square=True, cmap=plt.cm.Blues)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        matplotlib.rcParams.update({'font.size': 14})
        plt.savefig('%s_norm.png' % cm_path, pad_inches = 0, bbox_inches='tight')
        
        cm = confusion_matrix(y_true, y_pred)
        # Finding the annotations
        cm = cm.tolist()
        norm_cm = norm_cm.tolist()
        annot = [
            [("%d (%.2f)" % (c, nc)) for c, nc in zip(r, nr)]
            for r, nr in zip(cm, norm_cm)
        ]
        plt.figure(figsize = (10,7))
        sn.heatmap(norm_df_cm, annot=annot, fmt='', cbar=False, square=True, cmap=plt.cm.Blues)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        matplotlib.rcParams.update({'font.size': 14})
        plt.savefig('%s.png' % cm_path, pad_inches = 0, bbox_inches='tight')
        print (cm)
        
        METRICS_FILE='%s_metrics.txt'%cm_path

        accuracy = np.sum(y_true == y_pred) / len(y_true)
        
        with open(METRICS_FILE,'a') as file:
            file.write("Accuracy: %.5f \n" % accuracy)
            if len(labels)==2:
                file.write(classification_report(y_true,y_pred,labels=[0,1],target_names=labels))
            else:
                file.write(classification_report(y_true,y_pred,labels=[0,1,2],target_names=labels))

        
#         print ("Accuracy: %.5f" % accuracy)




def evaluate(test_list=TEST_LIST,CKPT_PATH_G=None, CKPT_PATH_L=None, CKPT_PATH_F=None,
             cm_path=None,roc_path=None,bs=16,combine_pneumonia=True):
    
    if combine_pneumonia:
        N_CLASSES = 3
        CLASSES = ["Normal", "Pneumonia", "Covid"]
    else:
        N_CLASSES= 4
        CLASSES=["Normal","Bacterial","Viral","Covid"]
    
    
    print('********************load data********************')
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    test_dataset = ChestXrayDataSetTest(image_list_file=test_list,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize,
                                    ]),
                                       combine_pneumonia=combine_pneumonia,crop=True)
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=bs,
                             shuffle=False, num_workers=4, pin_memory=True)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if USE_GPU:
        Global_Branch_model = CovidAidAttend(combine_pneumonia).cuda()
        Local_Branch_model = CovidAidAttend(combine_pneumonia).cuda()
#         Crop_model = CovidAidAttend(combine_pneumonia).cuda()

        Fusion_Branch_model = Fusion_Branch(input_size=2048, output_size=N_CLASSES).cuda()
    else:
        Global_Branch_model = CovidAidAttend(combine_pneumonia)
        Local_Branch_model = CovidAidAttend(combine_pneumonia)
#         Crop_model = CovidAidAttend(combine_pneumonia)

        Fusion_Branch_model = Fusion_Branch(input_size=2048, output_size=N_CLASSES)
    

    
    
    
    if os.path.exists(CKPT_PATH_G):
        if USE_GPU:
            checkpoint = torch.load(CKPT_PATH_G)
        else:                
            checkpoint = torch.load(CKPT_PATH_G,map_location='cpu')
        Global_Branch_model.load_state_dict(checkpoint)
        print("=> loaded Global_Branch_model checkpoint")

    if os.path.exists(CKPT_PATH_L):
        if USE_GPU:
            checkpoint = torch.load(CKPT_PATH_L)
            
        else:
            checkpoint = torch.load(CKPT_PATH_L,map_location='cpu')
        Local_Branch_model.load_state_dict(checkpoint)
        print("=> loaded Local_Branch_model checkpoint")

    if os.path.exists(CKPT_PATH_F):
        if USE_GPU:
            checkpoint = torch.load(CKPT_PATH_F)
        
        else:
            checkpoint = torch.load(CKPT_PATH_F,map_location='cpu')
        Fusion_Branch_model.load_state_dict(checkpoint)
        print("=> loaded Fusion_Branch_model checkpoint")
            
    cudnn.benchmark = True
    print('******************** load model succeed!********************')

    print('******* begin testing!*********')
    
    test(Global_Branch_model, Local_Branch_model, Fusion_Branch_model,
         test_loader,len(test_dataset),'test',cm_path,roc_path,combine_pneumonia)
    
def tensor_imshow(inp, title=None,bbox=None,img_name=None,size=None,vis_dir=None, **kwargs):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # Mean and std for ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    
    if bbox is None:
        plt.imshow(inp, **kwargs)
        if title is not None:
            plt.title(title)
    else:
        rr,cc = rectangle_perimeter(bbox[:2],bbox[2:],shape=inp.shape)
#         plt.figure(figsize=(10, 5))
#         plt.axis('off')

#         plt.imshow(inp,**kwargs)
#         if title is not None:
#             plt.title(title)
        save_path=os.path.join(vis_dir,'vis_%s'%img_name[0])
#         print(save_path)
        rescaled= np.uint8(inp*255)
        color=np.array([255,0,0])
        set_color(rescaled,(rr,cc),color)
#         print(size)
        new_size =(480,480)

        img = Image.fromarray(rescaled).convert('RGB')
        img = img.resize(new_size)

        
        img.save(save_path)
        
#         plt.savefig(save_path)    

def visualize(img_dir= None,CKPT_PATH_G=None, CKPT_PATH_L=None, 
              CKPT_PATH_F=None, vis_dir=None, combine_pneumonia=True, binary_eval=True):
    
    if combine_pneumonia:
        N_CLASSES = 3
        CLASSES = ["Normal", "Pneumonia", "Covid"]
    else:
        N_CLASSES= 4
        CLASSES=["Normal","Bacterial","Viral","Covid"]
    
    print('********************load data********************')
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    
    
    test_dataset = CovidDataLoader(image_dir= img_dir,
                                   transform=transforms.Compose([
                                   transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   normalize,
                                   ]))
    
        
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=1,
                    shuffle=False, num_workers=8, pin_memory=True)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if USE_GPU:
        Global_Branch_model = CovidAidAttend(combine_pneumonia).cuda()
        Local_Branch_model = CovidAidAttend(combine_pneumonia).cuda()
        Fusion_Branch_model = Fusion_Branch(input_size=2048, output_size=N_CLASSES).cuda()
    else:
        Global_Branch_model = CovidAidAttend(combine_pneumonia)
        Local_Branch_model = CovidAidAttend(combine_pneumonia)
        Fusion_Branch_model = Fusion_Branch(input_size=2048, output_size=N_CLASSES)
    
    if os.path.exists(CKPT_PATH_G):
        if USE_GPU:
            checkpoint = torch.load(CKPT_PATH_G)
        else:                
            checkpoint = torch.load(CKPT_PATH_G,map_location='cpu')
        Global_Branch_model.load_state_dict(checkpoint)
        print("=> loaded Global_Branch_model checkpoint")

    if os.path.exists(CKPT_PATH_L):
        if USE_GPU:
            checkpoint = torch.load(CKPT_PATH_L)
            
        else:
            checkpoint = torch.load(CKPT_PATH_L,map_location='cpu')
        Local_Branch_model.load_state_dict(checkpoint)
        print("=> loaded Local_Branch_model checkpoint")

    if os.path.exists(CKPT_PATH_F):
        if USE_GPU:
            checkpoint = torch.load(CKPT_PATH_F)
        
        else:
            checkpoint = torch.load(CKPT_PATH_F,map_location='cpu')
        Fusion_Branch_model.load_state_dict(checkpoint)
        print("=> loaded Fusion_Branch_model checkpoint")
            
    cudnn.benchmark = True
    print('******************** load model succeed!********************')

    Global_Branch_model.eval()
    Local_Branch_model.eval()
    Fusion_Branch_model.eval()
    
    pred_global = torch.FloatTensor()
    pred_local = torch.FloatTensor()
    pred_fusion = torch.FloatTensor()
    pred_names=[]
    print("Generating Attention Maps")
    
    for i, (image,name,size) in tqdm(enumerate(test_loader), total=len(test_loader)):
        
        pred_names+=name
        if USE_GPU:
            input_var = torch.autograd.Variable(image.cuda())
        else:
            input_var = torch.autograd.Variable(image)
            
        output_global, fm_global, pool_global = Global_Branch_model(input_var)

        try :
            patchs_var, boundingList= Attention_gen_patchs(image, fm_global,'visualize')
        except ValueError:
            patchs_var, boundingList= Attention_gen_patchs(image, fm_global)

        output_local, _, pool_local = Local_Branch_model(patchs_var)
           
        output_fusion = Fusion_Branch_model(pool_global.data, pool_local.data)
        
        pred_global = torch.cat((pred_global,output_global.data.cpu()),0)
        pred_local = torch.cat((pred_local,output_local.data.cpu()),0)
        pred_fusion = torch.cat((pred_fusion,output_fusion.data.cpu()),0)
       
        tensor_imshow(image.squeeze(0),bbox=boundingList[0],img_name=name,size=size,vis_dir=vis_dir)
        
    if binary_eval:
        pred_global = torch.from_numpy(pred_converter(pred_global.numpy()))
        pred_local = torch.from_numpy(pred_converter(pred_local.numpy()))
        pred_fusion = torch.from_numpy(pred_converter(pred_fusion.numpy()))
    
    scoresG = []
    for p,  n in zip(pred_global.numpy(), pred_names):
        p = ["%.1f %%" % (i * 100) for i in p]
#         l = np.argmax(l)
        scoresG.append([n] + p)
    scoresL = []
    for p,  n in zip(pred_local.numpy(), pred_names):
        p = ["%.1f %%" % (i * 100) for i in p]
#         l = np.argmax(l)
        scoresL.append([n] + p)
    scoresF = []
    for p,  n in zip(pred_fusion.numpy(), pred_names):
        p = ["%.1f %%" % (i * 100) for i in p]
#         l = np.argmax(l)
        scoresF.append([n] + p)
    
    header=['Name', 'Normal', 'Bacterial', 'Viral', 'COVID-19']
    alignment="c"*5
    if combine_pneumonia:
        header = ['Name', 'Normal', 'Pneumonia', 'COVID-19']
        alignment = "c"*4
    if binary_eval:
        header=["Name", "Non-Covid", "Covid"]
    
    predsFile=os.path.join(vis_dir,'predsGlobal.txt')
    f=open(predsFile,"w")
    pprint(header,f)
    pprint(scoresG,f)
    f.close()
    
    predsFile=os.path.join(vis_dir,'predsLocal.txt')
    f=open(predsFile,"w")
    pprint(header,f)
    pprint(scoresL,f)
    f.close()
    
    predsFile=os.path.join(vis_dir,'predsFusion.txt')
    f=open(predsFile,"w")
    pprint(header,f)
    pprint(scoresF,f)
    f.close()
    
    
    print("Visualizations Generated at: %s" %vis_dir)

    


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['train', 'test','visualize'], required=True)
    parser.add_argument("--combine_pneumonia", action='store_true', default=False)
    parser.add_argument("--save", type=str, default='models/')
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--epochs", type=int,default=50)
    parser.add_argument("--resume",action='store_true',default=False)
    parser.add_argument("--ckpt_init",type=str,default='data/CovidXNet_transfered_3.pth.tar')
    parser.add_argument("--ckpt_G",type=str,default='models/Global_Best.pth')
    parser.add_argument("--ckpt_L",type=str,default='models/Local_Best.pth')
    parser.add_argument("--ckpt_F",type=str,default='models/Fusion_Best.pth')
    parser.add_argument("--cm_path", type=str, default='plots/cm')
    parser.add_argument("--roc_path", type=str, default='plots/roc')
    parser.add_argument("--img_dir", type=str)
    parser.add_argument("--visualize_dir", type=str, default=None)
    parser.add_argument("--freeze", action='store_true', default=False)



    
    args = parser.parse_args()
    
#     if not args.scratch and args.ckpt_G==None:
#         print("Specify the paths for indivisual models")
#         exit(-1)

#     print(args.resume)
        
    if args.mode=='train':
        if args.resume:
            train('init', args.ckpt_G, args.ckpt_L, args.ckpt_F, args.epochs,args.bs,True, 
                  args.save, args.combine_pneumonia, args.freeze)
        else:
            train(args.ckpt_init,None,None,None,args.epochs,args.bs,True,args.save,
                  args.combine_pneumonia, args.freeze)

    elif args.mode=='test':
        evaluate(TEST_LIST, args.ckpt_G, args.ckpt_L, args.ckpt_F, args.cm_path, args.roc_path, args.bs,
                 args.combine_pneumonia)
    
    else:
        visualize(args.img_dir, args.ckpt_G, args.ckpt_L, args.ckpt_F, args.visualize_dir, args.combine_pneumonia)





    


