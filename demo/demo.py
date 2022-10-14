import argparse
import os
import sys
import matplotlib.image as mpimg

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import dataloader
from models import DFFNet
import numpy as np
import skimage.filters as skf
from models.submodule import *
from PIL import Image

import  matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

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


model = DFFNet(clean=False,level=args.level, use_diff=0,cnnlayers=1)
model = nn.DataParallel(model)
model.cuda()

if args.loadmodel is not None:
    pretrained_dict = torch.load(args.loadmodel)
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if 'disp' not in k}
    model.load_state_dict(pretrained_dict['state_dict'],strict=False)
else:
    print('run with random init')
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
model.eval()

'''
read data from directory
'''
phone_img_path=r"C:\Users\lahir\code\trained_models\savedimgs"
out_path=r'C:\Users\lahir\code\trained_models\demoimages'
imgs=os.listdir(phone_img_path)
jpgs=[item for item in imgs if (item.split('.')[-1]=='jpg')]
diopters=list(np.sort(np.unique(np.array([float(item.split('_')[-1][:-4]) for item in jpgs]))))
diopters.reverse()
selected_files=[]

for i,d in enumerate(diopters):
    for item in jpgs:
        diop=float(item.split('_')[-1][:-4])
        if(diop==d):
            selected_files.append(item)
            im = Image.open(phone_img_path+'\\'+item)
            im.save(out_path+r'\000046_0'+str(i)+'All.tif', 'TIFF')
            break


'''
Load data using the FoD500 dataloader
'''
import importlib
import dataloader
importlib.reload(dataloader)
database=out_path+'\\'
FoD500_train, _ = dataloader.FoD500Loader(database, n_stack=5, scale=1)
FoD500_train =[FoD500_train]


TrainImgLoader = torch.utils.data.DataLoader(dataset=FoD500_train, num_workers=1, batch_size=1, shuffle=True, drop_last=True)
dataset_train = torch.utils.data.ConcatDataset(FoD500_train)


for batch_idx, (img_stack, gt_disp, blur_stack,foc_dist) in enumerate(dataset_train):
    print('in here')
    break
img_stack=torch.unsqueeze(img_stack,0)
#foc_dist=torch.unsqueeze(foc_dist,0)
#foc_dist=torch.tensor([[0.1,0.5,0.9,1.3,1.7]])
diopters=list(1/np.array(diopters))
foc_dist=torch.tensor([diopters])
foc_dist=foc_dist.float()


fdepth3,std3,cost3=model(img_stack,foc_dist)

rgb=img_stack.detach().cpu()[0,0,0,:,:].numpy()
plt.figure()
plt.imshow(rgb, cmap='gray',interpolation='nearest')
plt.savefig("C:\\Users\\lahir\\code\\trained_models\\gray.png")
plt.show() 


depthmap=fdepth3.detach().cpu()[0,0,:,:].numpy()
depthmap*=1.7
plt.figure()
plt.imshow(depthmap, interpolation='nearest')
plt.colorbar()
plt.savefig("C:\\Users\\lahir\\code\\trained_models\\depth.png")
plt.show()










