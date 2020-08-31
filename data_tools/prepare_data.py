"""
Script to prepare combined dataset
Class 0: Normal
Class 1: Bacterial Pneumonia
Class 2: Viral Pneumonia
Class 3: COVID-19
"""
import glob
import os
import argparse
from itertools import islice
import random 
from random import shuffle
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--combine_pneumonia", action='store_true', default=False)
parser.add_argument("--bsti", action='store_true', default=False)
parser.add_argument("--kgp_action", action='store_true', default=False)
# parser.add_argument("--aiims", action='store_true', default=False)
parser.add_argument("--bmcv", action='store_true', default=False)

args = parser.parse_args()

COVID19_DATA_PATH = "./data/covid19"
COVID19_IMGS_PATH = "./covid-chestxray-dataset"
BSTI_IMGS_PATH = "./BSTI"
KGP_IMGS_PATH='./IITKGP-Action/images/'
PNEUMONIDA_DATA_PATH = "./chest-xray-pneumonia"
RSNA_PATH="./rsna-pneumonia-detection-challenge"
# AIIMS_DATA_PATH = "./AIIMS_Final_data"
BMCV_IMGS_PATH='./BIMCV-COVID19/'
DATA_PATH = "./data"
# AIIMS_TRAIN_PATH='./data/aiimsTrain_new.txt'
# AIIMS_TEST_PATH ='./data/aiimsTest_new.txt'

# Assert that the data directories are present
check_list = [COVID19_DATA_PATH, COVID19_IMGS_PATH, PNEUMONIDA_DATA_PATH, DATA_PATH]
if args.bsti:
    check_list.append(BSTI_IMGS_PATH)
if args.kgp_action:
    check_list.append(KGP_IMGS_PATH)
if args.bmcv:
    check_list.append(BMCV_IMGS_PATH)
# if args.aiims:
#     check_list.append(AIIMS_DATA_PATH)
for d in check_list:
    try:
        assert os.path.isdir(d) 
    except:
        print ("Directory %s does not exists" % d)

def create_list (split):
    random.seed(4)

    assert split in ['train', 'test', 'val']

    l = []
    tempNormal=[]
    tempPneumonia=[]
    # Prepare list using kaggle pneumonia dataset
    for f in glob.glob(os.path.join(PNEUMONIDA_DATA_PATH, split, 'NORMAL', '*')):
        tempNormal.append((f, 0)) # Class 0

    for f in glob.glob(os.path.join(PNEUMONIDA_DATA_PATH, split, 'PNEUMONIA', '*')):
        if args.combine_pneumonia:
            tempPneumonia.append((f, 1)) # Class 1
        else:
            if 'bacteria' in f:
                tempPneumonia.append((f, 1)) # Class 1
            else:
                tempPneumonia.append((f, 2)) # Class 2
    
    shuffle(tempNormal)
    shuffle(tempPneumonia)
    Nidx = int(0.05 * len(tempNormal))
    Pidx = int(0.05 * len(tempPneumonia))
    normalUsed = tempNormal[:Nidx]
    pneumoniaUsed = tempPneumonia[:Pidx]
#     l+=normalUsed
#     l+=pneumoniaUsed
    
    # Add RSNA Pneumonia Data According to Split
    
    TRAIN_CSV = pd.read_csv(os.path.join(RSNA_PATH,"stage_2_train_labels.csv"))
    INFO_CSV = pd.read_csv(os.path.join(RSNA_PATH,"stage_2_detailed_class_info.csv"))
    
    new_df = pd.merge(INFO_CSV,TRAIN_CSV,on='patientId',sort=False)
    dfnormal = new_df[(new_df['Target']==0) & (new_df['class']=='Normal')]
    normalRSNA = dfnormal['patientId'].unique().tolist()
    
    dfpneumonia = new_df[new_df['Target']==1]
    pneumoniaRSNA = dfpneumonia['patientId'].unique().tolist()
    
    shuffle(normalRSNA)
    shuffle(pneumoniaRSNA)
    
    if split=='train':
        normalRSNA = normalRSNA[900:]
        pneumoniaRSNA= pneumoniaRSNA[900:]
    elif split=='val':
        normalRSNA = normalRSNA[:300]
        pneumoniaRSNA= pneumoniaRSNA[:300]
    else:
        normalRSNA = normalRSNA[300:900]
        pneumoniaRSNA= pneumoniaRSNA[300:900]
        
    trainNames = glob.glob("./rsna-pneumonia-detection-challenge/stage_2_train_images/*.jpg")
#     testNames = glob.glob("./rsna-pneumonia-detection-challenge/stage_2_test_images/*.jpg")
    
    for file in trainNames:
        name = file.split("/")[-1][:-4]
        if name in normalRSNA:
            normalUsed.append((file,0))
        elif name in pneumoniaRSNA:
            pneumoniaUsed.append((file,1))
        else:
            continue
    
    print("Number of Public Normal %s Images Used: %d" % (split,len(normalUsed)))
    print("Number of Public Pneumonia %s Images Used: %d" % (split,len(pneumoniaUsed)))
    
    l+=normalUsed        
    l+=pneumoniaUsed

    
    
    ############################################
    # Prepare list using covid dataset
    covid_file = os.path.join(COVID19_DATA_PATH, '%s_list.txt'%split)
    with open(covid_file, 'r') as cf:
        for f in cf.readlines():
            f = os.path.join(COVID19_IMGS_PATH, f.strip())
            if args.combine_pneumonia:
                l.append((f, 2)) # Class 2
            else:
                l.append((f, 3)) # Class 3
                
    # Prepare list using BSTI covid dataset
    if args.bsti:
        bsti_covid_file = os.path.join(COVID19_DATA_PATH, 'bsti_%s_list.txt'%split)
        with open(bsti_covid_file, 'r') as cf:
            for f in cf.readlines():
                f = os.path.join(BSTI_IMGS_PATH, f.strip())
                if args.combine_pneumonia:
                    l.append((f, 2)) # Class 2
                else:
                    l.append((f, 3)) # Class 3
    
    # Prepare list using IIT-KGP Action Group covid dataset
    if args.kgp_action:
        kgp_covid_file = os.path.join(COVID19_DATA_PATH, 'kgp_%s_list.txt'%split)
        with open(kgp_covid_file, 'r') as cf:
            for f in cf.readlines():
                f = os.path.join(KGP_IMGS_PATH, f.strip())
                if args.combine_pneumonia:
                    l.append((f, 2)) # Class 2
                else:
                    l.append((f, 3)) # Class 3
    
    # Prepare list using BIMCV dataset                
    if args.bmcv:
        bmcv_covid_file=os.path.join(COVID19_DATA_PATH,'bmcv_%s_list.txt'%split)
        with open(bmcv_covid_file,'r') as cf:
            for f in cf.readlines():
                f=os.path.join(BMCV_IMGS_PATH,f.strip())
                if args.combine_pneumonia:
                    l.append((f,2)) # Class 2
                else:
                    l.append((f,3)) # Class 3
                    
#     if args.aiims:
#         if args.combine_pneumonia:
#             aiims_normalList=[f for f in glob.glob(os.path.join(AIIMS_DATA_PATH,'Normal','AP','*'))]
#             aiims_PneumoniaList=[f for f in glob.glob(os.path.join(AIIMS_DATA_PATH,'Pneumonia','AP','*'))]
#             aiims_covidList=[f for f in glob.glob(os.path.join(AIIMS_DATA_PATH,'COVID','*'))]
            
#             Normal_train_test_split=[4,170]
#             Pneumonia_train_test_split=[0,110]
#             Covid_train_test_split=[537,220]
            
#             aiims_N_iter=iter(aiims_normalList)
#             Normal_train_test_list=[list(islice(aiims_N_iter,elem)) for elem in Normal_train_test_split]
            
#             aiims_P_iter=iter(aiims_PneumoniaList)
#             Pneumonia_train_test_list=[list(islice(aiims_P_iter,elem)) for elem in Pneumonia_train_test_split]
            
#             aiims_C_iter=iter(aiims_covidList)
#             Covid_train_test_list=[list(islice(aiims_C_iter,elem)) for elem in Covid_train_test_split]
            
#             if split=='train':
#                 for file in Normal_train_test_list[0]:
#                     l.append((file,0))
#                 for file in Pneumonia_train_test_list[0]:
#                     l.append((file,1))
                
#                 for file in Covid_train_test_list[0]:
#                     l.append((file,2))
            
#             elif split=='test':
#                 for file in Normal_train_test_list[1]:
#                     l.append((file,0))
#                 for file in Pneumonia_train_test_list[1]:
#                     l.append((file,1))
                
#                 for file in Covid_train_test_list[1]:
#                     l.append((file,2))
            
            
#         else:
#             print("Can't use AIIMS data as Viral and Bacterial Pneumonia data for it hasn't been defined.")
#             raise ValueError
    
#     print("Adding AIIMS data as well")
#     aiims_train=[]
#     aiims_test=[]
    
#     with open(AIIMS_TEST_PATH, "r") as f:
#             for line in f:
#                 items = line.split()
#                 image_name = ' '.join(items[:-1])

#                 label = int(items[-1])
#                 if os.path.exists(image_name):
#                     aiims_test.append((image_name, label))
#                 else:
#                     continue
    
#     with open(AIIMS_TRAIN_PATH, "r") as f:
#             for line in f:
#                 items = line.split()
#                 image_name = ' '.join(items[:-1])

#                 label = int(items[-1])
#                 if os.path.exists(image_name):
#                     aiims_train.append((image_name, label))
#                 else:
#                     continue
                    
#     if split=='train':
#         l+=aiims_train
#         print("Added {} AIIMS train images".format(len(aiims_train)))
#     elif split=='val':
#         l+=aiims_test
#         print("Added {} AIIMS val images".format(len(aiims_test)))
#     else:
#         pass
    
    # Print Final List Statistics
    
    print('Total %s Images = %5d'%(split,len(l)))
    
    with open(os.path.join(DATA_PATH, '%s_NEW2.txt'%split), 'w') as f:
        for item in l:
            f.write("%s %d\n" % item)
            
# def create_test_list (split):
#     assert split in ['test']

#     l = []
#     # Prepare list using AIIMS dataset
#     for f in glob.glob(os.path.join(AIIMS_DATA_PATH, 'Normal', '*')):
#         l.append((f, 0)) # Class 0

#     for f in glob.glob(os.path.join(AIIMS_DATA_PATH, 'Pneumonia', '*')):
#         if args.combine_pneumonia:
#             l.append((f, 1)) # Class 1
#         else:
#             if 'bacteria' in f:
#                 l.append((f, 1)) # Class 1
#             else:
#                 l.append((f, 2)) # Class 2
        
                
#     for f in glob.glob(os.path.join(AIIMS_DATA_PATH, 'Covid', '*')):
#         if args.combine_pneumonia:
#             l.append((f, 2)) # Class 2
#         else:
#             l.append((f, 3)) # Class 3

#     with open(os.path.join(DATA_PATH, 'aiims_%s.txt'%split), 'w') as f:
#         for item in l:
#             f.write("%s %d\n" % item)

    
    
for split in ['train', 'test', 'val']:
    create_list(split)
        
        