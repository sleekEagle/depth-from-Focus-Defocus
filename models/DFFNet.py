from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.submodule import *
import pdb
from models.featExactor2 import FeatExactor
from models.AENet import AENet


# Ours-FV (use_diff=0) and Ours-DFV (use_diff=1) model
        
class depthNet(nn.Module):
    def __init__(self,n_fs,cnnlayers):
        super(depthNet, self).__init__()
        #self.conv1=nn.Conv2d(n_fs*2, 1, kernel_size=1, stride=1, padding=0)
        #self.sig=nn.Sigmoid()
        self.layers = nn.ModuleList()
        if(cnnlayers==1):
            self.layers.append(nn.Conv2d(n_fs*2, 1, kernel_size=1, stride=1, padding=0))
        else:
            self.layers.append(nn.Conv2d(n_fs*2, 32, kernel_size=1, stride=1, padding=0))
            for i in range(cnnlayers-1):
                #print("cnnlayers "+str(cnnlayers)+" i "+str(i))
                if(i==(cnnlayers-2)):
                    self.layers.append(nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0))
                    break
                self.layers.append(nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0))

    def forward(self, x, focal_dist=None):
        #print('confnet in : '+str(x.shape))
        #input=torch.cat([x,focal_dist],dim=1)
        for layer in self.layers:
            x=F.relu(layer(x))
        return x
 
        

class DFFNet(nn.Module):
    def __init__(self,clean,level=1,use_diff=1,cnnlayers=1):
        super(DFFNet, self).__init__()

        self.clean = clean
        self.feature_extraction = FeatExactor()
        self.level = level
        self.sig=nn.Sigmoid()

        self.use_diff = use_diff
        assert level >= 1 and level <= 4
        assert use_diff == 0 or use_diff == 1

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

        # reg
        #self.disp_reg = disparityregression(1)        

        self.depthnet=depthNet(5,cnnlayers)

    def diff_feat_volume1(self, vol):
        vol_out = vol[:,:, :-1] - vol[:, :, 1:]
        return torch.cat([vol_out, vol[:,:, -1:]], dim=2) # last elem is  vol[:,:, -1] - 0

    def forward(self, stack, focal_dist):
        b, n, c, h, w = stack.shape
        input_stack = stack.reshape(b*n, c, h , w)

        conv4, conv3, conv2, conv1  = self.feature_extraction(input_stack)

        # conv3d take b, c, d, h, w
        _vol4, _vol3, _vol2, _vol1  = conv4.reshape(b, n, -1, h//32, w//32).permute(0, 2, 1, 3, 4), \
                                 conv3.reshape(b, n, -1, h//16, w//16).permute(0, 2, 1, 3, 4),\
                                 conv2.reshape(b, n, -1, h//8, w//8).permute(0, 2, 1, 3, 4),\
                                 conv1.reshape(b, n, -1, h//4, w//4).permute(0, 2, 1, 3, 4)
                                 
        conv4ae, conv3ae, conv2ae, conv1ae  = conv4.reshape(b, n, -1, h//32, w//32), \
                                 conv3.reshape(b, n, -1, h//16, w//16),\
                                 conv2.reshape(b, n, -1, h//8, w//8),\
                                 conv1.reshape(b, n, -1, h//4, w//4)
        
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
    
        foc_ar=focal_dist.unsqueeze(dim=2).unsqueeze(dim=3).\
        repeat_interleave(cost3.shape[2],dim=2).\
        repeat_interleave(cost3.shape[3],dim=3)
        fdepth3,ddepth,fddepth3,std3=-1,-1,-1,-1

        #fdepth3,std3=self.disp_reg(F.softmax(cost3,1),focal_dist, uncertainty=True)
        
        depthinp=torch.cat([cost3,foc_ar],dim=1)
        fdepth3=self.depthnet(depthinp)
        
        input_stack = stack.reshape(b*n, c, h , w)
        input_stack = stack.reshape(b*n, c, h , w)
        
        # different output based on level for training
        fstacked=[fdepth3]
        #fdstacked=[fddepth3]
        stds=[std3]
        cost_stacked=[cost3]

        #if training the model
        if self.training :
            if self.level >= 2:
                std4=-1
                cost4=F.interpolate(cost4, [h, w], mode='bilinear')
                cost_stacked.append(cost4)
                #fdepth4,std4=self.disp_reg(F.softmax(cost4,1),focal_dist, uncertainty=True)        
                depthinp=torch.cat([cost4,foc_ar],dim=1)
                fdepth4=self.depthnet(depthinp)
                stds.append(std4)
                fstacked.append(fdepth4)
                
                if self.level >=3 :    
                    std5=-1
                    cost5=F.interpolate(cost5, [h, w], mode='bilinear')
                    #cost5 = F.interpolate((cost5).unsqueeze(1), [focal_dist.shape[1], h, w], mode='trilinear').squeeze(1)
                    cost_stacked.append(cost5)
                    #fdepth5,std5=self.disp_reg(F.softmax(cost5,1),focal_dist, uncertainty=True)
                    depthinp=torch.cat([cost4,foc_ar],dim=1)
                    fdepth5=self.depthnet(depthinp)
                    stds.append(std5)
                    fstacked.append(fdepth5)

                    if self.level >=4 :
                        std6=-1
                        #cost6 = F.interpolate((cost6).unsqueeze(1), [focal_dist.shape[1], h, w], mode='trilinear').squeeze(1)
                        cost6=F.interpolate(cost6, [h, w], mode='bilinear')
                        cost_stacked.append(cost6)
                        depthinp=torch.cat([cost6,foc_ar],dim=1)
                        fdepth6=self.depthnet(depthinp)
                        #fdepth6,std6=self.disp_reg(F.softmax(cost6,1),focal_dist, uncertainty=True)
                        stds.append(std6)
                        fstacked.append(fdepth6)
                        
            return fstacked,stds,cost_stacked
        #if evaluating the model
        else:
            return fdepth3,std3,cost3

'''
model=DFFNet(clean=False,level=4)
img_stack=torch.rand((5,5,3,224,224))
foc_stack=torch.rand((5,5))
regstacked,stds,cost=model(img_stack,foc_stack)
'''
#stack=torch.rand(4,5,3,224,224)
#focal_dist=torch.rand(1,5)
#out=model(stack,fostack=torch.rand(4,5,3,224,224)cal_dist)
