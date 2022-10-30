import os
import numpy as np
from torch.utils.data import Dataset
import torchvision
import torch
from PIL import Image

from torchvision import transforms
import random
import numbers
import OpenEXR
from os import listdir, mkdir
from os.path import isfile, join, isdir
import cv2

# code adopted from https://github.com/soyers/ddff-pytorch/blob/master/python/ddff/dataproviders/datareaders/FocalStackDDFFH5Reader.py

def read_dpt(img_dpt_path):
    im = cv2.imread(img_dpt_path,-1)
    return im

foc_dist=[2.0, 0.89, 0.57, 3.29, 1.1, 0.66, 0.75, 1.4, 10.0, 0.49]
foc_dist_needed=[1.9,0.58,0.66,1.32]

class ImageDataset(torch.utils.data.Dataset):
    """Focal place dataset."""
    #max_dpt=2.8398,min_dpt=0.1000
    def __init__(self,img_list, dpth_list,  transform_fnc=None, flag_shuffle=False, data_ratio=0,
                 flag_inputs=[True, False], flag_outputs=[False, True], f_number=0.1,max_dpt=1.5):

        self.transform_fnc = transform_fnc
        self.flag_shuffle = flag_shuffle

        self.flag_rgb = flag_inputs[0]
        self.flag_coc = flag_inputs[1]

        self.data_ratio = data_ratio

        self.flag_out_coc = flag_outputs[0]
        self.flag_out_depth = flag_outputs[1]
                
        self.max_dpt=max_dpt


        self.img_mean = np.array([0.485, 0.456, 0.406]).reshape([1,1,3])#[0.278, 0.250, 0.227]
        self.img_std= np.array([0.229, 0.224, 0.225]).reshape([1,1,3])#[0.185, 0.178, 0.178]

        self.guassian_kernel =  (35,35) # large blur kernel for pad image

        ##### Load all images
        self.imglist_all = img_list
        self.imglist_dpt = dpth_list


    def __len__(self):
        return int(len(self.imglist_dpt))

    def dpth2disp(self, dpth):
        disp = 1 / dpth
        disp[dpth==0] = 0
        return disp

    def __getitem__(self, idx):
        ##### Read and process an image
        idx_dpt = int(idx)        
        img_dpt = read_dpt(self.imglist_dpt[idx_dpt])

        idxdir=self.imglist_dpt[idx].split('\\')[-2]
        imgdirs=[f.split('\\')[-2] for f in self.imglist_all]
        imgpaths=[self.imglist_all[i] for i,item in enumerate(imgdirs) if item==idxdir]
        foc_dist=[float(f.split('\\')[-1].split('_')[-1][:-4]) for f in imgpaths]
        '''
        foc_dist=[2.0, 0.89, 0.57, 3.29, 1.1, 0.66, 0.75, 1.4, 10.0, 0.49]
        foc_dist.sort()
        imgpaths=['C:\\Users\\lahir\\fstack_data\\data_processed\\10_19_2022_15_39_37\\10_19_2022_15_39_38.764175_2.0.png', 'C:\\Users\\lahir\\fstack_data\\data_processed\\10_19_2022_15_39_37\\10_19_2022_15_39_39.016790_0.89.png', 'C:\\Users\\lahir\\fstack_data\\data_processed\\10_19_2022_15_39_37\\10_19_2022_15_39_39.551168_0.57.png', 'C:\\Users\\lahir\\fstack_data\\data_processed\\10_19_2022_15_39_37\\10_19_2022_15_39_39.817852_3.29.png', 'C:\\Users\\lahir\\fstack_data\\data_processed\\10_19_2022_15_39_37\\10_19_2022_15_39_40.075090_1.1.png', 'C:\\Users\\lahir\\fstack_data\\data_processed\\10_19_2022_15_39_37\\10_19_2022_15_39_40.328020_0.66.png', 'C:\\Users\\lahir\\fstack_data\\data_processed\\10_19_2022_15_39_37\\10_19_2022_15_39_41.119419_0.75.png', 'C:\\Users\\lahir\\fstack_data\\data_processed\\10_19_2022_15_39_37\\10_19_2022_15_39_41.904925_1.4.png', 'C:\\Users\\lahir\\fstack_data\\data_processed\\10_19_2022_15_39_37\\10_19_2022_15_39_43.753240_10.0.png', 'C:\\Users\\lahir\\fstack_data\\data_processed\\10_19_2022_15_39_37\\10_19_2022_15_39_44.801376_0.49.png']
        imgpaths=["a", "b", "c", "d", "e", "f", "g", "h", "i","j"]
        '''
        imgpaths = [x for _,x in sorted(zip(foc_dist,imgpaths),reverse=True)]
        foc_dist=sorted(foc_dist,reverse=True)

        #img_dpt=img_dpt/self.max_dpt
        mat_dpt = img_dpt.copy()[:, :, np.newaxis]
        
        # add RGB, CoC, Depth inputs
        mats_input = []
        mats_output = np.zeros((256, 256, 0))
        foc_selected=[]

        # load existing image
        for i in range(len(imgpaths)):
            fdist=float(imgpaths[i].split('\\')[-1].split('_')[-1][:-4])
            im = Image.open(imgpaths[i])
            img_all = np.array(im)
            # img Norm
            mat_all = img_all.copy() / 255.
            mat_all = (mat_all - self.img_mean) / self.img_std
            mats_input.append(mat_all)  
            foc_selected.append(fdist)
    
        mats_input=np.stack(mats_input)        
        sample = {'input': mats_input, 'output': mat_dpt}
        if self.transform_fnc:
            sample = self.transform_fnc(sample)
        #devide depth by 1000 to get m
        return sample['input'],sample['output']/1000,(1/torch.tensor(foc_dist)),2

class ToTensor(object):
    def __call__(self, sample):
        mats_input,mats_output = sample['input'], sample['output']
        mats_input=mats_input.transpose((0, 3, 1, 2))
        mats_output=mats_output.transpose((2, 0, 1))
        mats_output=mats_output.astype(np.float32)
        # print(mats_input.shape, mats_output.shape)
        return {'input': torch.from_numpy(mats_input).float(),
                'output': torch.from_numpy(mats_output).float()}

class RandomCrop(object):
    """ Randomly crop images
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        inputs,target = sample['input'], sample['output']
        n, h, w, _ = inputs.shape
        th, tw = self.size
        if w < tw: tw=w
        if h < th: th=h

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        inputs=inputs[:, y1: y1 + th,x1: x1 + tw]
        target=target[y1: y1 + th,x1: x1 + tw]
        return {'input':inputs,
                'output':target}


class RandomFilp(object):
    """ Randomly crop images
    """

    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def __call__(self, sample):
        inputs,target = sample['input'], sample['output']

        # hori filp
        if np.random.binomial(1, self.ratio):
            inputs=inputs[:,:, ::-1]
            target=target[:,::-1]

        # vert flip
        if np.random.binomial(1, self.ratio):
            inputs=inputs[:, ::-1]
            target=target[::-1]
        return {'input': np.ascontiguousarray(inputs), 'output': np.ascontiguousarray(target)}

def MobileKinectLoader(data_dir):
    dirs=listdir(data_dir)
    dirs.sort()
    train_dirs=dirs[:200]
    val_dirs=dirs[200:]

    dpth_train_list=[join(data_dir,d,f) for d in train_dirs for f in listdir(join(data_dir,d)) if isfile(join(data_dir,d,f)) and f.split('.')[0]=="depth"]
    dpth_val_list=[join(data_dir,d,f) for d in val_dirs for f in listdir(join(data_dir,d)) if isfile(join(data_dir,d,f)) and f.split('.')[0]=="depth"]

    img_train_list=[join(data_dir,d,f) for d in train_dirs for f in listdir(join(data_dir,d)) if isfile(join(data_dir,d,f)) and f.split('.')[0]!="depth"]
    img_val_list=[join(data_dir,d,f) for d in val_dirs for f in listdir(join(data_dir,d)) if isfile(join(data_dir,d,f)) and f.split('.')[0]!="depth"]

    img_train_list.sort()
    dpth_train_list.sort()

    img_val_list.sort()
    dpth_val_list.sort()

    train_transform = transforms.Compose([
                        RandomCrop(224),
                        RandomFilp(0.5),
                        ToTensor()])
 
    dataset_train = ImageDataset(img_list=img_train_list, dpth_list=dpth_train_list,
                                 transform_fnc=train_transform)

    val_transform = transforms.Compose([
                        RandomCrop(224),
                        ToTensor()])

    dataset_valid = ImageDataset(img_list=img_val_list, dpth_list=dpth_val_list,
                                 transform_fnc=val_transform)

    return dataset_train, dataset_valid

'''
data_dir='C:\\Users\\lahir\\fstack_data\\data_processed'
dataset_train, dataset_validation = MobileKinectLoader(data_dir)
TrainImgLoader = torch.utils.data.DataLoader(dataset=dataset_train, num_workers=0, batch_size=2, shuffle=True, drop_last=True)
#gtlist=torch.empty(0,1,224,224)

gtlist=torch.empty(0,10,3,224,224)
for batch_idx, (img_stack, gt_disp,foc_dist,dataset) in enumerate(TrainImgLoader):
    print(img_stack.shape)
    continue
    #gtlist=torch.cat((gtlist,img_stack),0)
'''
