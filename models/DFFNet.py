from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.submodule import *
from models.featExactor2 import FeatExactor


# Ours-FV (use_diff=0) and Ours-DFV (use_diff=1) model

def gets2(tensor):
    std=torch.std(tensor,axis=1,keepdim=True).repeat_interleave(tensor.shape[1],1)
    mean=torch.mean(tensor,axis=1,keepdim=True).repeat_interleave(tensor.shape[1],1)
    far=torch.abs((tensor-mean)/std)
    outlier=far<1.6
    clean=torch.where(outlier,tensor,mean)
    s2=1/torch.mean(clean,axis=1,keepdim=True)
    return s2

class DFFNet(nn.Module):
    #disp_mode : how to get distance from focal_score/blur
        #0 - disparity regression from DFV paper
        #1 - our linear NN
        #2 - our CNN
    def __init__(self,clean,level=1,use_diff=0):
        super(DFFNet, self).__init__()

        self.clean = clean
        self.feature_extraction = FeatExactor()
        self.level = level
        self.use_diff=use_diff

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

    def forward(self, stack, focal_dist):
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
        print('cost3:'+str(cost3.shape))
        fdepth3=gets2(cost3)
        fstacked.append(fdepth3)
        print('fdepth:'+str(fdepth3.shape))
        #print('fdepth:'+str(fdepth.shape))

        #if training the model
        if self.training :
            if self.level >= 2:
                std4=-1
                cost4=F.interpolate(cost4, [h, w], mode='bilinear')
                cost_stacked.append(cost4)
                fdepth4=gets2(cost4)
                fstacked.append(fdepth4)
                #fdepth4,std4=self.disp_reg(F.softmax(cost4,1),focal_dist, uncertainty=True)        
                
                if self.level >=3 :    
                    std5=-1
                    cost5=F.interpolate(cost5, [h, w], mode='bilinear')
                    cost_stacked.append(cost5)
                    fdepth5=gets2(cost5)
                    fstacked.append(fdepth5)
                    #fdepth5,std5=self.disp_reg(F.softmax(cost5,1),focal_dist, uncertainty=True)
                    if self.level >=4 :
                        std6=-1
                        cost6=F.interpolate(cost6, [h, w], mode='bilinear')
                        cost_stacked.append(cost6)
                        fdepth6=gets2(cost6)
                        fstacked.append(fdepth6)
                        #fdepth6,std6=self.disp_reg(F.softmax(cost6,1),focal_dist, uncertainty=True)
                        
            return fstacked,cost_stacked
        #if evaluating the model
        else:
            return fdepth3,cost3

'''
model=DFFNet(clean=0)
stack=torch.rand(4,5,3,224,224)
focal_dist=torch.rand(1,5)
out=model(stack,focal_dist)
'''


