from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.submodule import *
from models.featExactor2 import FeatExactor


# Ours-FV (use_diff=0) and Ours-DFV (use_diff=1) model
'''
import numpy as np
import matplotlib.pyplot as plt
s2=0.8
s1=np.arange(0.6,2.0,0.3)
val=np.abs(1-s1/s2)

1/(np.diff(val)/np.diff(s1))

plt.plot(val)
plt.show()

tensor=torch.from_numpy(val).view(1,-1,1,1).repeat_interleave(20,0).repeat_interleave(224,-2).repeat_interleave(224,-1)
s1t=torch.from_numpy(s1).view(1,-1,1,1).repeat_interleave(20,0).repeat_interleave(224,-2).repeat_interleave(224,-1)
focal_dist=torch.from_numpy(s1).view(1,-1).repeat_interleave(20,0)
focal_dist.shape
'''

class depthNet(nn.Module):
    def __init__(self,n_fs,cnnlayers):
        super(depthNet, self).__init__()
        self.layers = nn.ModuleList()
        if(cnnlayers==1):
            self.layers.append(nn.Conv2d(n_fs*2, 1, kernel_size=1, stride=1, padding=0))
        else:
            self.layers.append(nn.Conv2d(n_fs*2, 32, kernel_size=1, stride=1, padding=0))
            for i in range(cnnlayers-1):
                if(i==(cnnlayers-2)):
                    self.layers.append(nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0))
                    break
                self.layers.append(nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        for layer in self.layers:
            x=F.relu(layer(x))
        return x

def gets2(tensor,focal_dist):
    tensor=tensor+torch.abs(torch.randn_like(tensor))*0.1
    s1t=focal_dist.view(focal_dist.shape[0],focal_dist.shape[1],1,1) 
    s1t=torch.repeat_interleave(s1t,tensor.shape[-2],dim=2)
    s1t=torch.repeat_interleave(s1t,tensor.shape[-1],dim=3)

    tdiff=torch.diff(tensor,dim=1)
    s1diff=torch.diff(s1t,dim=1)
    #print(tensor[0,:,0,0])
    #print(tdiff[0,:,0,0])
    #print(s1diff[0,:,0,0])

    s2est=torch.abs(tdiff/s1diff)
    #print(s2est[0,:,0,0])

    std=torch.std(s2est,axis=1,keepdim=True).repeat_interleave(tdiff.shape[1],1)
    mean=torch.mean(s2est,axis=1,keepdim=True).repeat_interleave(tdiff.shape[1],1)
    far=torch.abs((s2est-mean)/std)
    clean=(far<1.0)*(s2est>0)
    nvals=torch.sum(clean,axis=1,keepdim=True)
    #print('nvals: '+str(nvals[0,:,0,0]))
    #print(nvals.shape)

    s2=1/torch.sum(s2est*clean,dim=1,keepdim=True)/nvals
    #print(s2.shape)
    #print(s2[0,:,0,0])
    #print("***********")
    return s2

class DFFNet(nn.Module):
    def __init__(self,clean,level=1,use_diff=0,numdatasets=3,cnnlayers=1):
        super(DFFNet, self).__init__()

        self.clean = clean
        self.feature_extraction = FeatExactor()
        self.level = level
        self.use_diff=use_diff
        self.numdatasets=numdatasets

        self.cameraparam = torch.nn.ParameterDict()
        '''
        define k_pix*N/f**2 and f for each camers
        k_pix = pixels per mm 
        N - f number 
        f - focal dist in mm
        '''
        self.cameraparam["camera{}".format(0)] = torch.nn.Parameter(data=torch.Tensor([1.439,9.3e-3]), requires_grad=False)
        self.cameraparam["camera{}".format(1)] = torch.nn.Parameter(data=torch.Tensor([1.5,9.3e-3]), requires_grad=True)
        self.cameraparam["camera{}".format(2)] = torch.nn.Parameter(data=torch.Tensor([1.5,9.3e-3]), requires_grad=True)
            
        self.depthnet=depthNet(5,cnnlayers)
        print(self.depthnet)

        assert level >= 1 and level <= 4

        if level == 1:
            self.decoder3 = decoderBlock(2,16,16, stride=(1,1,1),up=False, nstride=1)
        elif level == 2:
            self.decoder3 = decoderBlock(2,32,32, stride=(1,1,1),up=False, nstride=1)
            self.decoder4 =  decoderBlock(2,32,32, up=True)
        elif level == 3:
            self.decoder3 = decoderBlock(2, 32, 32, stride=(1, 1, 1), up=False, nstride=1)
            self.decoder4 = decoderBlock(2, 64, 32, up=True)
            self.decoder5 = decoderBlock(2, 64, 64, up=True, pool=True)
        else:
            self.decoder3 = decoderBlock(2, 32, 32, stride=(1, 1, 1), up=False, nstride=1)
            self.decoder4 = decoderBlock(2, 64, 32,  up=True)
            self.decoder5 = decoderBlock(2, 128, 64, up=True, pool=True)
            self.decoder6 = decoderBlock(2, 128, 128, up=True, pool=True)

    def diff_feat_volume1(self, vol):
        vol_out = vol[:,:, :-1] - vol[:, :, 1:]
        return torch.cat([vol_out, vol[:,:, -1:]], dim=2) # last elem is  vol[:,:, -1] - 0

    def forward(self, stack, focal_dist,dataset):
        b, n, c, h, w = stack.shape
        input_stack = stack.reshape(b*n, c, h , w)

        conv4, conv3, conv2, conv1  = self.feature_extraction(input_stack)

        _vol4, _vol3, _vol2, _vol1  = conv4.reshape(b, n, -1, h//32, w//32).permute(0, 2, 1, 3, 4), \
                                 conv3.reshape(b, n, -1, h//16, w//16).permute(0, 2, 1, 3, 4),\
                                 conv2.reshape(b, n, -1, h//8, w//8).permute(0, 2, 1, 3, 4),\
                                 conv1.reshape(b, n, -1, h//4, w//4).permute(0, 2, 1, 3, 4)

        
        if self.use_diff == 1:
            vol4, vol3, vol2, vol1 = self.diff_feat_volume1(_vol4), self.diff_feat_volume1(_vol3),\
                                     self.diff_feat_volume1(_vol2), self.diff_feat_volume1(_vol1)
        else:
            vol4, vol3, vol2, vol1 =  _vol4, _vol3, _vol2, _vol1

        if self.level == 1:
            _, cost3 = self.decoder3(vol1)

        elif self.level == 2:
            feat4_2x, cost4 = self.decoder4(vol2)
            feat3 = torch.cat((feat4_2x, vol1), dim=1)
            _, cost3 = self.decoder3(feat3)

        elif self.level == 3:
            feat5_2x, cost5 = self.decoder5(vol3)
            feat4 = torch.cat((feat5_2x, vol2), dim=1)

            feat4_2x, cost4 = self.decoder4(feat4)
            feat3 = torch.cat((feat4_2x, vol1), dim=1)
            _, cost3 = self.decoder3(feat3)

        else:
            feat6_2x, cost6 = self.decoder6(vol4)
            feat5 = torch.cat((feat6_2x, vol3), dim=1)
            feat5_2x, cost5 = self.decoder5(feat5)
            feat4 = torch.cat((feat5_2x, vol2), dim=1)

            feat4_2x, cost4 = self.decoder4(feat4)
            feat3 = torch.cat((feat4_2x, vol1), dim=1)
            _, cost3 = self.decoder3(feat3)
        
        
        cost3=F.interpolate(cost3, [h, w], mode='bilinear')
 
        input_stack = stack.reshape(b*n, c, h , w)
        input_stack = stack.reshape(b*n, c, h , w)
        
        # different output based on level for training
        #fdstacked=[fddepth3]
        cost_stacked=[cost3]
        fstacked=[]
        camconstlist=[]

        for i in range(self.numdatasets):
            a=torch.repeat_interleave(self.cameraparam["camera{}".format(i)][0],dataset.shape[0])
            a[(dataset!=i).nonzero(as_tuple=True)[0]]=1
            a=a.view(-1,1,1,1)
            a=torch.repeat_interleave(a,focal_dist.shape[1],dim=1)
            a=torch.abs(a)
            f=torch.repeat_interleave(self.cameraparam["camera{}".format(i)][1],dataset.shape[0]).view(-1,1)
            f=torch.repeat_interleave(f,focal_dist.shape[1],dim=1)
            f=torch.abs(f)
            fdf=(focal_dist-f)
            fdf[(dataset!=i).nonzero(as_tuple=True)[0],:]=1
            fdf=fdf.view(fdf.shape[0],fdf.shape[1],1,1)
            fdf=fdf.view(fdf.shape[0],fdf.shape[1],1,1)
            res=a*fdf
            #print(a[:,:,0,0]) 
            #print(fdf[:,:,0,0])  
            #print("*****************")
            camconstlist.append(res)
        val=cost3
        #print('cost3 :::::::')
        #print(torch.min(cost3))
        for i in range(self.numdatasets):
            val*=camconstlist[i]
        foc_ar=focal_dist.unsqueeze(dim=2).unsqueeze(dim=3).\
        repeat_interleave(cost3.shape[2],dim=2).\
        repeat_interleave(cost3.shape[3],dim=3)
        #print('val : '+str(val.shape))
        depthinp=torch.cat([val,foc_ar],dim=1)
        #print('depthinp : '+str(depthinp.shape))
        depth3=self.depthnet(depthinp)
        fstacked.append(depth3)
        '''
        max_f,_=torch.max(focal_dist,dim=1,keepdim=True)
        max_f=max_f.view(-1,1,1,1)
        max_f=torch.repeat_interleave(max_f,depth_scaled.shape[-1],-1)
        max_f=torch.repeat_interleave(max_f,depth_scaled.shape[-2],-2)
        depth3=depth_scaled*max_f
        '''
        #if training the model
        if self.training :
            if self.level >= 2:
                std4=-1
                cost4=F.interpolate(cost4, [h, w], mode='bilinear')
                cost_stacked.append(cost4)
                val=cost4
                for i in range(self.numdatasets):
                    val*=camconstlist[i]
                depthinp=torch.cat([val,foc_ar],dim=1)
                depth4=self.depthnet(depthinp)
                fstacked.append(depth4)
                #fdepth4,std4=self.disp_reg(F.softmax(cost4,1),focal_dist, uncertainty=True)        
                
                if self.level >=3 :    
                    std5=-1
                    cost5=F.interpolate(cost5, [h, w], mode='bilinear')
                    cost_stacked.append(cost5)
                    val=cost5
                    for i in range(self.numdatasets):
                        val*=camconstlist[i]
                    depthinp=torch.cat([val,foc_ar],dim=1)
                    depth5=self.depthnet(depthinp)
                    fstacked.append(depth5)
                    #fdepth5,std5=self.disp_reg(F.softmax(cost5,1),focal_dist, uncertainty=True)
                    if self.level >=4 :
                        std6=-1
                        cost6=F.interpolate(cost6, [h, w], mode='bilinear')
                        cost_stacked.append(cost6)
                        val=cost6
                        for i in range(self.numdatasets):
                            val*=camconstlist[i]
                        depthinp=torch.cat([val,foc_ar],dim=1)
                        depth6=self.depthnet(depthinp)
                        fstacked.append(depth6)
                        #fdepth6,std6=self.disp_reg(F.softmax(cost6,1),focal_dist, uncertainty=True)
                        
            return fstacked,1,cost_stacked
        #if evaluating the model
        else:
            return depth3,1,cost3

'''
model=DFFNet(clean=0,level=4)
stack=torch.rand(6,5,3,224,224)
focal_dist=torch.tensor([0.1,0.2,0.3,0.4,0.5]).view(1,-1)
focal_dist=torch.repeat_interleave(focal_dist,6,0)
dataset=torch.tensor([0,1,2,1,1,0], dtype=torch.long)
out=model(stack,focal_dist,dataset)
'''
'''
for n,p in model.named_parameters():
    if('camera' in n):
         print(n + '  ' + str(p)) 

cameraparam = torch.nn.ParameterDict()
cameraparam["camera{}".format(0)] = torch.nn.Parameter(data=torch.Tensor([1.4394136,9.3]), requires_grad=True)
cameraparam["camera{}".format(1)] = torch.nn.Parameter(data=torch.Tensor([1.0,8.5]), requires_grad=True)
cameraparam["camera{}".format(2)] = torch.nn.Parameter(data=torch.Tensor([1.0,8.5]), requires_grad=False)

i=0
cost=torch.ones((6,5,20,20))
a=torch.repeat_interleave(cameraparam["camera{}".format(0)][0],dataset.shape[0])
a[(dataset!=i).nonzero(as_tuple=True)[0]]=1
a=a.view(-1,1,1,1)
a=torch.repeat_interleave(a,focal_dist.shape[1],dim=1)

f=torch.repeat_interleave(cameraparam["camera{}".format(0)][1],dataset.shape[0]).view(-1,1)
f=torch.repeat_interleave(f,focal_dist.shape[1],dim=1)
fdf=(focal_dist-f)
fdf[(dataset!=i).nonzero(as_tuple=True)[0],:]=1
fdf=fdf.view(fdf.shape[0],fdf.shape[1],1,1)
fdf=fdf.view(fdf.shape[0],fdf.shape[1],1,1)
res=a*fdf
print(res[:,:,0,0])


depthnet=depthNet(5,3)
cost=torch.ones((4,5,20,20))
focal_dist=torch.tensor([0.1,0.2,0.3,0.4,0.5]).view(1,-1,1,1)
focal_dist=torch.repeat_interleave(focal_dist,4,0)
focal_dist=torch.repeat_interleave(focal_dist,20,-1)
focal_dist=torch.repeat_interleave(focal_dist,20,-2)

depth_scaled=depthnet(cost)
max_f,_=torch.max(focal_dist,dim=1,keepdim=True)
max_f=max_f.view(-1,1,1,1)
max_f=torch.repeat_interleave(max_f,depth_scaled.shape[-1],-1)
max_f=torch.repeat_interleave(max_f,depth_scaled.shape[-2],-2)
depth=depth_scaled*max_f
'''

