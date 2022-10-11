import argparse
import cv2
from models import DFFNet
import numpy as np
import os
import skimage.filters as skf
import time
from models.submodule import *

import  matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import torchvision



parser = argparse.ArgumentParser(description='DFVDFF')
parser.add_argument('--data_path', default='/data/DFF/my_ddff_trainVal.h5',help='test data path')
parser.add_argument('--loadmodel', default="C:\\Users\\lahir\\code\\trained_models\\best.tar", help='model path')
parser.add_argument('--outdir', default='./DDFF12/',help='output dir')

parser.add_argument('--max_disp', type=float ,default=0.28, help='maxium disparity')
parser.add_argument('--min_disp', type=float ,default=0.02, help='minium disparity')

parser.add_argument('--stack_num', type=int ,default=5, help='num of image in a stack, please take a number in [2, 10], change it according to the loaded checkpoint!')
parser.add_argument('--use_diff',type=int, default=0, choices=[0,1], help='if use differential images as input, change it according to the loaded checkpoint!')

parser.add_argument('--level', type=int, default=4, help='num of layers in network, please take a number in [1, 4]')
parser.add_argument('--fuse', default=0, type=int, choices=[0,1,2], help='how to fuse defocus cues and focus scores, 0: only focus (equivalent to DFV paper),  1: Only defocus based, 2: final depth=(defocus+focus)/2')
args = parser.parse_args()


model = DFFNet(clean=False,level=args.level,use_diff=args.use_diff,fuse=args.fuse)
model = nn.DataParallel(model)
model.cuda()

if args.loadmodel is not None:
    pretrained_dict = torch.load(args.loadmodel)
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if 'disp' not in k}
    model.load_state_dict(pretrained_dict['state_dict'],strict=False)
else:
    print('run with random init')
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

img=torch.rand((1,5,3,224,224))
focal_dist=torch.rand((1,5))

model(img,focal_dist)



