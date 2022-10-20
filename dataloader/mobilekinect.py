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

class ImageDataset(torch.utils.data.Dataset):
    """Focal place dataset."""
    #max_dpt=2.8398,min_dpt=0.1000
    def __init__(self, root_dir, img_list, dpth_list,  transform_fnc=None, flag_shuffle=False, img_num=5, data_ratio=0,
                 flag_inputs=[True, False], flag_outputs=[False, True], focus_dist=[0.1,.15,.3,0.7,1.5], f_number=0.1, 
                 scale=1,max_dpt=1.5):
        self.root_dir = root_dir
        self.transform_fnc = transform_fnc
        self.flag_shuffle = flag_shuffle

        self.flag_rgb = flag_inputs[0]
        self.flag_coc = flag_inputs[1]

        self.img_num = img_num
        self.data_ratio = data_ratio

        self.flag_out_coc = flag_outputs[0]
        self.flag_out_depth = flag_outputs[1]

        self.focus_dist = [f/max_dpt for f in focus_dist]
        
        self.max_n_stack = 5
        self.dpth_scale = scale
        
        self.camera=CameraLens.CameraLens(2.9*1e-3,f_number=1)
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
        img_dpt = read_dpt(self.root_dir + self.imglist_dpt[idx_dpt])
        img_dpt=img_dpt/self.max_dpt

        foc_dist = self.focus_dist.copy()
        mat_dpt = img_dpt.copy()[:, :, np.newaxis]

        img_num = min(self.max_n_stack, self.img_num)
        ind = idx * img_num

        num_list = list(range(self.max_n_stack))

        # add RGB, CoC, Depth inputs
        mats_input = []
        blur_list=[]
        mats_output = np.zeros((256, 256, 0))

        # load existing image
        pad_lst = []
        pad_focs = []
        for i in range(self.max_n_stack):
            if self.flag_rgb:
                im = Image.open(self.root_dir + self.imglist_all[ind + num_list[i]])
                img_all = np.array(im)
                # img Norm
                mat_all = img_all.copy() / 255.
                mat_all = (mat_all - self.img_mean) / self.img_std
                mats_input.append(mat_all)

                # pad invalid img in the beginning or the end, keep diff consistent
                if self.img_num > self.max_n_stack and len(pad_lst) == 0:
                    for j in range(self.img_num-self.max_n_stack):
                        pad_lst.append(cv2.GaussianBlur(mat_all, self.guassian_kernel, 0))
                        pad_focs.append(0)
            #calc CoC images
            blur=self.camera.get_coc(self.focus_dist[i],img_dpt)
            blur = np.clip(blur, 0, 1.272e-4) / 1.272e-4
            blur_list.append(blur)

        mats_input = pad_lst + mats_input
        foc_dist = pad_focs + foc_dist


        mats_input=np.stack(mats_input)
        blur_list=np.stack(blur_list)
        
        if img_num < self.max_n_stack:
            if len( self.imglist_all) > 100: # train
                rand_idx = np.random.choice(self.max_n_stack, img_num,
                                            replace=False)  # this will shuffle order as well
                rand_idx = np.sort(rand_idx)
            else:
                rand_idx = np.linspace(0, self.max_n_stack, img_num)

            mats_input = mats_input[rand_idx]
            foc_dist = [foc_dist[i] for i in rand_idx]
            blur_list=blur_list[rand_idx]

        if self.flag_out_depth:
            mats_output = np.concatenate((mats_output,(mat_dpt)), axis=2) # first 5 is COC last is depth  self.dpth2disp
        
        sample = {'input': mats_input, 'output': mats_output,'blur':(blur_list)}

        if self.transform_fnc:
            sample = self.transform_fnc(sample)
        return sample['input'],sample['output'],sample['blur'],(torch.tensor(foc_dist))
        #print('before return ')
        #return torch.rand(2,3),torch.rand(3,4)

class ToTensor(object):
    def __call__(self, sample):
        mats_input,mats_output,coc = sample['input'], sample['output'],sample['blur']
        mats_input=mats_input.transpose((0, 3, 1, 2))
        coc=coc.transpose((0, 1, 2))
        mats_output=mats_output.transpose((2, 0, 1))
        # print(mats_input.shape, mats_output.shape)
        return {'input': torch.from_numpy(mats_input).float(),
                'output': torch.from_numpy(mats_output).float(),
               'blur':torch.from_numpy(coc).float()}


class RandomCrop(object):
    """ Randomly crop images
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        inputs,target,coc = sample['input'], sample['output'],sample['blur']
        n, h, w, _ = inputs.shape
        th, tw = self.size
        if w < tw: tw=w
        if h < th: th=h

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        inputs=inputs[:, y1: y1 + th,x1: x1 + tw]
        coc=coc[:, y1: y1 + th,x1: x1 + tw]
        return {'input':inputs,
                'output':target[y1: y1 + th,x1: x1 + tw],
                'blur':coc}


class RandomFilp(object):
    """ Randomly crop images
    """

    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def __call__(self, sample):
        inputs,target,coc = sample['input'], sample['output'],sample['blur']

        # hori filp
        if np.random.binomial(1, self.ratio):
            inputs=inputs[:,:, ::-1]
            target=target[:,::-1]
            coc=coc[:,:, ::-1]

        # vert flip
        if np.random.binomial(1, self.ratio):
            inputs=inputs[:, ::-1]
            target=target[::-1]
            coc=coc[:, ::-1]

        return {'input': np.ascontiguousarray(inputs), 'output': np.ascontiguousarray(target),'blur': np.ascontiguousarray(coc)}



def FoD500Loader(data_dir, n_stack=5, scale=1, focus_dist=[0.1,.15,.3,0.7,1.5]):
    data_dir=r'C:\Users\lahir\fstack_data\data_processed'
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
    dataset_train = ImageDataset(root_dir=data_dir, img_list=img_train_list, dpth_list=dpth_train_list,
                                 transform_fnc=train_transform, img_num=n_stack,  focus_dist=focus_dist, scale=scale)

    val_transform = transforms.Compose([ToTensor()])
    dataset_valid = ImageDataset(root_dir=data_dir, img_list=img_val_list, dpth_list=dpth_val_list,
                                 transform_fnc=val_transform, img_num=n_stack, focus_dist=focus_dist, scale=scale)


    return dataset_train, dataset_valid
