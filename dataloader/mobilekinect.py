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
from dataloader import CameraLens


# code adopted from https://github.com/soyers/ddff-pytorch/blob/master/python/ddff/dataproviders/datareaders/FocalStackDDFFH5Reader.py

img_dpt_path=r'C:\Users\lahir\fstack_data\data_processed\10_19_2022_15_39_37\depth.png'
img_dpt_path=r'C:\Users\lahir\focusdata\fs_6\fs_6\000000Dpt.exr'
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
                
        self.camera=CameraLens(2.9*1e-3,f_number=1)
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
        print(foc_dist)
        
        #img_dpt=img_dpt/self.max_dpt
        mat_dpt = img_dpt.copy()[:, :, np.newaxis]
        print(mat_dpt.shape)
        
        # add RGB, CoC, Depth inputs
        mats_input = []
        blur_list=[]
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
            
            
            #calc CoC images
            blur=self.camera.get_coc(fdist,img_dpt)
            blur = np.clip(blur, 0, 1.272e-4) / 1.272e-4
            blur_list.append(blur)
    
        mats_input=np.stack(mats_input)
        print(mats_input.shape)
        print('foc selected : ',foc_selected)
        
        blur_list=np.stack(blur_list)
        print('blue shape:',str(blur_list.shape))
                
        sample = {'input': mats_input, 'output': mat_dpt,'blur':blur_list}
        print('before trans...')
        if self.transform_fnc:
            sample = self.transform_fnc(sample)
        #devide depth by 1000 to get m
        return sample['input'],sample['output']/1000,sample['blur'],foc_dist


class ToTensor(object):
    def __call__(self, sample):
        mats_input,mats_output,blur = sample['input'], sample['output'],sample['blur']
        mats_input=mats_input.transpose((0, 3, 1, 2))
        blur=blur.transpose((0, 1, 2))
        mats_output=mats_output.transpose((2, 0, 1))
        mats_output=mats_output.astype(np.float32)
        # print(mats_input.shape, mats_output.shape)
        return {'input': torch.from_numpy(mats_input).float(),
                'output': torch.from_numpy(mats_output).float(),
               'blur':torch.from_numpy(blur).float()}

class RandomCrop(object):
    """ Randomly crop images
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        inputs,target,blur = sample['input'], sample['output'],sample['blur']
        n, h, w, _ = inputs.shape
        th, tw = self.size
        if w < tw: tw=w
        if h < th: th=h

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        inputs=inputs[:, y1: y1 + th,x1: x1 + tw]
        blur=blur[:, y1: y1 + th,x1: x1 + tw]
        target=target[y1: y1 + th,x1: x1 + tw]
        return {'input':inputs,
                'output':target,
                'blur':blur}


class RandomFilp(object):
    """ Randomly crop images
    """

    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def __call__(self, sample):
        inputs,target,blur = sample['input'], sample['output'],sample['blur']

        # hori filp
        if np.random.binomial(1, self.ratio):
            inputs=inputs[:,:, ::-1]
            target=target[:,::-1]
            blur=blur[:,:, ::-1]

        # vert flip
        if np.random.binomial(1, self.ratio):
            inputs=inputs[:, ::-1]
            target=target[::-1]
            blur=blur[:, ::-1]
        return {'input': np.ascontiguousarray(inputs), 'output': np.ascontiguousarray(target),'blur': np.ascontiguousarray(blur)}


data_dir=r'C:\Users\lahir\fstack_data\data_processed'

def MobileKinectLoader(data_dir):
    dirs=listdir(data_dir)
    dirs.sort()
    train_dirs=dirs[:3]
    val_dirs=dirs[3:]

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
    it=iter(dataset_valid)
    a,b,c,d=next(it)

    val_transform = transforms.Compose([ToTensor()])
    dataset_valid = ImageDataset(img_list=img_val_list, dpth_list=dpth_val_list,
                                 transform_fnc=val_transform)


    return dataset_train, dataset_valid

path=r'C:\Users\lahir\Documents\CPR Quality Data\CoolTerm Capture 2022-10-20 15-08-48.txt'
