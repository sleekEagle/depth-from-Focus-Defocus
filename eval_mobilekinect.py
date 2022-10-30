import argparse
import cv2
from models import DFFNet
import numpy as np
import os
import skimage.filters as skf
import time
from models.submodule import *
import torch.optim as optim
import  matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import torchvision
# dataloader
from dataloader import MobileKinectLoader


parser = argparse.ArgumentParser(description='DFVDFF')
parser.add_argument('--data_path', default='C:\\Users\\lahir\\fstack_data\\data_processed\\',help='test data path')
parser.add_argument('--loadmodel', default='C:\\Users\\lahir\\focusdata\\best.tar', help='model path')
parser.add_argument('--outdir', default='./DDFF12/',help='output dir')
parser.add_argument('--stack_num', type=int ,default=5, help='num of image in a stack, please take a number in [2, 10], change it according to the loaded checkpoint!')
parser.add_argument('--use_diff',type=int, default=0, choices=[0,1], help='if use differential images as input, change it according to the loaded checkpoint!')
parser.add_argument('--level', type=int, default=4, help='num of layers in network, please take a number in [1, 4]')
parser.add_argument('--lr', type=float, default=0.01,  help='learning rate')
parser.add_argument('--blur', type=float, default=0,  help='loss weight for blur')
args = parser.parse_args()


# construct the model
model = DFFNet(clean=False,level=args.level,use_diff=args.use_diff)
model = nn.DataParallel(model)
model.cuda()
ckpt_name = os.path.basename(os.path.dirname(args.loadmodel))
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

if args.loadmodel is not None:
    pretrained_dict = torch.load(args.loadmodel)
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if 'disp' not in k}
    model.load_state_dict(pretrained_dict['state_dict'],strict=False)
else:
    print('run with random init')
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
#freeze all weights except camera2 paramerers

for param in model.parameters():
    param.requires_grad = False
for n,p in model.named_parameters():
    if('camera2' in n):
        p.requires_grad=True

data_dir=r'C:\Users\lahir\fstack_data\data_processed'
dataset_train, dataset_validation = MobileKinectLoader(data_dir)
trainLoader = torch.utils.data.DataLoader(dataset=dataset_train, num_workers=0, batch_size=10, shuffle=True)
valiLoader = torch.utils.data.DataLoader(dataset=dataset_validation, num_workers=0, batch_size=1, shuffle=True)


for n,p in model.named_parameters():
    if('camera' in n):
         print(n + '  ' + str(p) + "  " + str(p.requires_grad))

# =========== Train func. =========
def train(img_stack,gt_disp, foc_dist,dataset):
    model.train()
    img_stack=Variable(torch.FloatTensor(img_stack))
    gt_disp=Variable(torch.FloatTensor(gt_disp))
    img_stack,gt_disp,foc_dist=img_stack.cuda(),gt_disp.cuda(),foc_dist.cuda()
    
    #---------
    max_val = torch.where(foc_dist>=100, torch.zeros_like(foc_dist), foc_dist) # exclude padding value
    min_val = torch.where(foc_dist<=0, torch.ones_like(foc_dist)*10, foc_dist)  # exclude padding value
    mask = (gt_disp >= min_val.min(dim=1)[0].view(-1,1,1,1)) & (gt_disp <= max_val.max(dim=1)[0].view(-1,1,1,1)) #
    mask_tiled=torch.repeat_interleave(mask,repeats=5,dim=1)
    mask_blur=torch.repeat_interleave(mask,repeats=1,dim=1)
    mask.detach_()
    mask_tiled.detach_()
    mask_blur.detach_()
    #----

    optimizer.zero_grad()
    beta_scale = 1 # smooth l1 do not have beta in 1.6, so we increase the input to and then scale back -- no significant improve according to our trials
    regstacked,stds,cost= model(img_stack, foc_dist,dataset)

    gt_disp_=torch.repeat_interleave(gt_disp,repeats=img_stack.shape[1],dim=1)
    foc_dist_=foc_dist.view(foc_dist.shape[0],foc_dist.shape[1],1,1)
    foc_dist_=torch.repeat_interleave(foc_dist_,repeats=gt_disp.shape[-1],dim=-1)
    foc_dist_=torch.repeat_interleave(foc_dist_,repeats=gt_disp.shape[-2],dim=-2)
    blur=torch.abs(1-foc_dist_/gt_disp_)

    dloss,bloss=0,0
    lvl_w=[8./15, 4./15, 2./15, 1./15]
    for i in range(len(regstacked)):
        _cur_floss = F.smooth_l1_loss(regstacked[i][mask] * beta_scale, gt_disp[mask]* beta_scale, reduction='none') / beta_scale
        dloss = dloss + lvl_w[i] * _cur_floss.mean()
        if(args.blur>0):
            _cur_bloss=F.mse_loss(cost[i][mask_tiled],blur[mask_tiled],reduction='none').mean()
            bloss = bloss + lvl_w[i] * _cur_bloss.mean()
    
    loss=dloss+args.blur*bloss
    loss.backward()   
    torch.nn.utils.clip_grad_norm_(model.module.parameters(), max_norm=0.5)
    optimizer.step()
    vis={}
    vis['pred']=regstacked[0].detach().cpu()
    vis['mask']=mask.type(torch.float).detach().cpu()
   
    del regstacked,cost
    return loss.data, dloss.item(),vis

def valid(img_stack_in,disp,foc_dist,dataset):
    model.eval()
    img_stack_in=Variable(torch.FloatTensor(img_stack_in))
    gt_disp=Variable(torch.FloatTensor(disp))
    img_stack,gt_disp,foc_dist=img_stack_in.cuda(),gt_disp.cuda(),foc_dist.cuda()
    
    #---------
    mask = gt_disp > 0
    mask.detach_()
    #----
    with torch.no_grad():
        regdepth,stds,cost= model(img_stack, foc_dist,dataset)
        loss=(F.mse_loss(regdepth[mask] , gt_disp[mask] , reduction='mean')) # use MSE loss for val
    vis = {}
    vis['mask'] = mask.type(torch.float).detach().cpu()
    vis["pred"]=regdepth.detach().cpu()
    
    return loss, vis

min_f,max_f=0.3,1.3
epoch=1
total_iters=0
for epoch in range(700):
    total_train_loss = 0
    for batch_idx, (img_stack, gt_disp,foc_dist,dataset) in enumerate(trainLoader):
        f=(foc_dist[0,:]<max_f)&(foc_dist[0,:]>min_f)
        ind=f.nonzero()[:,0]
        foc_dist=foc_dist[:,ind]
        img_stack=img_stack[:,ind,:,:,:]
        if(foc_dist.shape[-1]!=5):
            continue
        loss,dloss,viz=train(img_stack,gt_disp,foc_dist,dataset)
        total_train_loss += loss
        total_iters += 1
        torch.cuda.synchronize()
        if total_iters %10 == 0:
            torch.cuda.synchronize()
            print('epoch %d:  %d/ %d loss = %.6f, dloss = %.6f' % (epoch, batch_idx, len(trainLoader), loss,dloss))
    # Vaild
    if epoch % 5 == 0:
        total_val_loss = 0
        for batch_idx, (img_stack, gt_disp,foc_dist,dataset) in enumerate(valiLoader):
            f=(foc_dist[0,:]<max_f)&(foc_dist[0,:]>min_f)
            ind=f.nonzero()[:,0]
            foc_dist=foc_dist[:,ind]
            img_stack=img_stack[:,ind,:,:,:]
            if(foc_dist.shape[-1]!=5):
                continue

            with torch.no_grad():
                start_time = time.time()
                val_loss,viz=valid(img_stack,gt_disp,foc_dist,dataset)
            total_val_loss += val_loss
            if batch_idx %10 == 0:
                torch.cuda.synchronize()
                print('[val] epoch %d : %d/%d val_loss = %.6f' % (epoch, batch_idx, len(valiLoader), val_loss))
        avg_val_loss = total_val_loss / len(valiLoader)
        print('[val] avg_val_loss %.6f' %avg_val_loss)

        #print camera param 
        for n,p in model.named_parameters():
            if('camera' in n):
                print(n + '  ' + str(p))   
       
       
'''
        break
train(img_stack,gt_disp,foc_dist,dataset)

regstacked,stds,cost= model(img_stack, foc_dist,dataset)


numbers = [1, 2, 3]
letters = ['a', 'b', 'c']
zipped = zip(numbers, letters)
'''




