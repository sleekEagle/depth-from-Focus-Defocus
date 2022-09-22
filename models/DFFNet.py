from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from models.submodule import *
import pdb
from models.featExactor2 import FeatExactor


# Ours-FV (use_diff=0) and Ours-DFV (use_diff=1) model
x=torch.rand((1,5))
focal_dist=torch.rand((1,5))
input=torch.cat([x,focal_dist],dim=1)
lin1=nn.Linear(10,1)
lin1(input)


class disparityNetLin(nn.Module):
    def __init__(self,n_fs):
        super(disparityNetLin, self).__init__()
        self.n_fs=n_fs
        self.lin1=nn.Linear(n_fs*2,1)

    def forward(self, x, focal_dist=None):
        input=torch.cat([x,focal_dist],dim=1)
        out=self.lin1(input)
        return out
        
dmodel=disparityNetLin(5)
dmodel(x,focal_dist)
       
        

class DFFNet(nn.Module):
    #disp_mode : how to get distance from focal_score/blur
        #0 - disparity regression from DFV paper
        #1 - our linear NN
        #2 - our CNN
    def __init__(self,clean,level=1,use_diff=1,fuse=0,disp_mode=0):
        super(DFFNet, self).__init__()

        self.clean = clean
        self.feature_extraction = FeatExactor()
        self.level = level
        self.fuse=fuse
        self.sig=nn.Sigmoid()
        self.disp_mode=disp_mode

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
        self.disp_reg = disparityregression(1)
        self.disp_net=disparityNetLin(5)
        
        #depth
        dlayers=[]
        for i in range(len(dchlist)-1):
            dlayers.append(nn.Conv2d(dchlist[i],dchlist[i+1],1,1,bias=False))
            #dlayers.append(nn.BatchNorm2d(dchlist[i+1]))
            if(dpool):
                dlayers.append(nn.MaxPool2d((3,3),(3,3)))
            #dlayers.append(nn.Upsample(size=(224,224)))
        if(dsigmoid):
            dlayers.append(self.sig)
	
         
       # dlayers.append(nn.BatchNorm2d(1))
        self.depthNet=nn.Sequential(*dlayers)
        print(self.depthNet)

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

        #d=_vol1.shape[0]*_vol1.shape[1]*_vol1.shape[2]
        #n=_vol1.reshape(d,_vol1.shape[3],_vol1.shape[4])
        
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
        fdepth3,ddepth3,fddepth3,std3=-1,-1,-1,-1
        if(self.fuse==0 or self.fuse==2):
            if(self.disp_mode==0):
                fdepth3,std3=self.disp_reg(F.softmax(cost3,1),focal_dist, uncertainty=True)
            elif(self.disp_mode==1):
                fdepth3=self.disp_net()
        if(self.fuse==2):
            fddepth3=(fdepth3+ddepth3)*0.5
        #for defocus-based method, calculate depth from blur
        if(self.fuse==1 or self.fuse==2):
            blur3=1./cost3 -1
            foc_ar=focal_dist.unsqueeze(dim=2).unsqueeze(dim=3).\
            repeat_interleave(blur3.shape[2],dim=2).\
            repeat_interleave(blur3.shape[3],dim=3)
            bf3=torch.cat((blur3,foc_ar),dim=1)
            ddepth3=self.depthNet(bf3)
            ddepth3=F.interpolate(ddepth3,[h,w],mode='bilinear')
        input_stack = stack.reshape(b*n, c, h , w)
        input_stack = stack.reshape(b*n, c, h , w)
        
        # different output based on level for training
        fstacked=[fdepth3]
        dstacked=[ddepth3]
        fdstacked=[fddepth3]
        stds=[std3]
        cost_stacked=[cost3]
        
        #if training the model
        if self.training :
            if self.level >= 2:
                cost4=self.sig(cost4)
                cost4=F.interpolate(cost4, [h, w], mode='bilinear')
                cost_stacked.append(cost4)
                if(self.fuse==1 or self.fuse==2):
                    blur4=1./cost4 -1
                    bf4=torch.cat((blur4,foc_ar),dim=1)
                    ddepth4=self.depthNet(bf4)
                    dstacked.append(ddepth4)
                if(self.fuse==0 or self.fuse==2):
                    fdepth4,std4=self.disp_reg(F.softmax(cost4, 1), focal_dist, uncertainty=True)  
                    stds.append(std4)
                    fstacked.append(fdepth4)
                if(self.fuse==2):
                    pred4=(fdepth4+ddepth4)*0.5
                    fdstacked.append(pred4)
                
                #total_depth=torch.cat((pred4,depth),dim=1)        
                #pred4=total_depth*conf
                
                if self.level >=3 :
                    cost5=self.sig(cost5)
                    cost5 = F.interpolate((cost5).unsqueeze(1), [focal_dist.shape[1], h, w], mode='trilinear').squeeze(1)
                    cost_stacked.append(cost5)
                    if(self.fuse==1 or self.fuse==2):
                        blur5=1./cost5 -1
                        bf5=torch.cat((blur5,foc_ar),dim=1)
                        ddepth5=self.depthNet(bf5)
                        dstacked.append(ddepth5)
                    if(self.fuse==0 or self.fuse==2):
                        fdepth5,std5=self.disp_reg(F.softmax(cost5, 1), focal_dist, uncertainty=True)  
                        stds.append(std5)
                        fstacked.append(fdepth5)
                    if(self.fuse==2):
                        pred5=(fdepth5+ddepth5)*0.5
                        fdstacked.append(pred5)

                    if self.level >=4 :
                        cost6=self.sig(cost6)
                        cost6 = F.interpolate((cost6).unsqueeze(1), [focal_dist.shape[1], h, w], mode='trilinear').squeeze(1)
                        cost_stacked.append(cost6)
                        if(self.fuse==1 or self.fuse==2):
                            blur6=1./cost6 -1
                            bf6=torch.cat((blur6,foc_ar),dim=1)
                            ddepth6=self.depthNet(bf6)
                            dstacked.append(ddepth6)
                        if(self.fuse==0 or self.fuse==2):
                            fdepth6,std6=self.disp_reg(F.softmax(cost6, 1), focal_dist, uncertainty=True)  
                            stds.append(std6)
                            fstacked.append(fdepth6)
                        if(self.fuse==2):
                            pred6=(fdepth6+ddepth6)*0.5
                            fdstacked.append(pred6)
                        
            return fstacked,dstacked,fdstacked,stds,cost_stacked
        #if evaluating the model
        else:
            print('std3')
            print(std3)
            if(self.fuse==0 or self.fuse==2):
                std3=torch.squeeze(std3)
            return fdepth3,ddepth3,fddepth3,std3,cost3
