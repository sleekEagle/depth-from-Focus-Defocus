from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from models.submodule import *
import pdb
from models.featExactor2 import FeatExactor
from models.AENet import AENet


# Ours-FV (use_diff=0) and Ours-DFV (use_diff=1) model

class disparityNetCNN(nn.Module):
    def __init__(self,n_fs):
        super(disparityNetCNN, self).__init__()
        self.n_fs=n_fs
        self.lin1=nn.Linear(n_fs*2,1)
        self.conv1=nn.Conv2d(2*n_fs, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x, focal_dist=None):
        #print('x shape ' +str(x.shape))
        #print('foca shape '+str(focal_dist.shape))
        input=torch.cat([x,focal_dist],dim=1)
        out=self.conv1(input)
        return out
        
#dmodel=disparityNetLin(5)
#dmodel(x,focal_dist)
       
        

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
	
        self.dispNet=disparityNetCNN(5)
        print(self.dispNet)
        self.aenet=AENet(out_dim=1,nstacks=5,num_filter=16)

    def diff_feat_volume1(self, vol):
        vol_out = vol[:,:, :-1] - vol[:, :, 1:]
        return torch.cat([vol_out, vol[:,:, -1:]], dim=2) # last elem is  vol[:,:, -1] - 0

    def forward(self, stack, focal_dist):
        b, n, c, h, w = stack.shape
        input_stack = stack.reshape(b*n, c, h , w)

        conv4, conv3, conv2, conv1  = self.feature_extraction(input_stack)
        #print('conv1: '+str(conv1.shape))
        #print('conv2: '+str(conv2.shape))
        #print('conv3: '+str(conv3.shape))
        #print('conv4: '+str(conv4.shape))
        # conv3d take b, c, d, h, w
        _vol4, _vol3, _vol2, _vol1  = conv4.reshape(b, n, -1, h//32, w//32).permute(0, 2, 1, 3, 4), \
                                 conv3.reshape(b, n, -1, h//16, w//16).permute(0, 2, 1, 3, 4),\
                                 conv2.reshape(b, n, -1, h//8, w//8).permute(0, 2, 1, 3, 4),\
                                 conv1.reshape(b, n, -1, h//4, w//4).permute(0, 2, 1, 3, 4)
                                 
        conv4ae, conv3ae, conv2ae, conv1ae  = conv4.reshape(b, n, -1, h//32, w//32), \
                                 conv3.reshape(b, n, -1, h//16, w//16),\
                                 conv2.reshape(b, n, -1, h//8, w//8),\
                                 conv1.reshape(b, n, -1, h//4, w//4)
        #to be used as input to the AENet
        down=[conv4ae,conv3ae,conv2ae,conv1ae]


        #print('conv1ae: '+str(conv1ae.shape))
        #print('conv2ae: '+str(conv2ae.shape))
        #print('conv3ae: '+str(conv3ae.shape))
        #print('conv4ae: '+str(conv4ae.shape))           

        #print('vol1: '+str(_vol1.shape))
        #print('vol2: '+str(_vol2.shape))
        #print('vol3: '+str(_vol3.shape))
        #print('vol4: '+str(_vol4.shape))        
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
        
        #print('cost3: '+str(cost3.shape))
        #print('cost4: '+str(cost4.shape))
        #print('cost5: '+str(cost5.shape))
        #print('cost6: '+str(cost6.shape))
        
        cost3=F.interpolate(cost3, [h, w], mode='bilinear')
        #print('cost3 '+str(cost3.shape))
        #print('f ar '+str(focal_dist.shape))
        if(self.disp_mode==1 or self.fuse==1 or self.fuse==2):
            foc_ar=focal_dist.unsqueeze(dim=2).unsqueeze(dim=3).\
            repeat_interleave(cost3.shape[2],dim=2).\
            repeat_interleave(cost3.shape[3],dim=3)
            #print('focal ar shape '+str(foc_ar.shape))
        fdepth3,ddepth,fddepth3,std3=-1,-1,-1,-1
        if(self.fuse==0 or self.fuse==2):
            if(self.disp_mode==0):
                fdepth3,std3=self.disp_reg(F.softmax(cost3,1),focal_dist, uncertainty=True)
                #print('fdepth3:'+str(fdepth3.shape))
            elif(self.disp_mode==1):
                fdepth3=self.dispNet(cost3,foc_ar)
                #print('fdepth3_net:'+str(fdepth3.shape))
            if(self.fuse==2):
                fddepth3=(fdepth3+ddepth3)*0.5
        #for defocus-based method, calculate depth from blur
        if(self.fuse==1 or self.fuse==2):
            blur3=cost3
            #print('blur3 ' +str(blur3.shape))
            #print('foc_ar '+str(foc_ar.shape))
            #for item in down:
            #    print(item.shape)
            ddepth=self.aenet(blur3,down,5,foc_ar)
            #print('depth net out ' +str(ddepth.shape))
            ddepth=F.interpolate(ddepth,[h,w],mode='bilinear')
        input_stack = stack.reshape(b*n, c, h , w)
        input_stack = stack.reshape(b*n, c, h , w)
        
        # different output based on level for training
        fstacked=[fdepth3]
        fdstacked=[fddepth3]
        stds=[std3]
        cost_stacked=[cost3]
        
        #if training the model
        if self.training :
            if self.level >= 2:
                cost4=F.interpolate(cost4, [h, w], mode='bilinear')
                cost_stacked.append(cost4)
                if(self.fuse==0 or self.fuse==2):
                    if(self.disp_mode==0):
                        fdepth4,std4=self.disp_reg(F.softmax(cost4,1),focal_dist, uncertainty=True)
                        stds.append(std4)
                        #print('fdepth4:'+str(fdepth4.shape))
                    elif(self.disp_mode==1):
                        fdepth4=self.dispNet(cost4,foc_ar)
                        #print('fdepth4_net:'+str(fdepth4.shape))
                    fstacked.append(fdepth4)
                if(self.fuse==2):
                    pred4=(fdepth4+ddepth)*0.5
                    fdstacked.append(pred4)
                
                #total_depth=torch.cat((pred4,depth),dim=1)        
                #pred4=total_depth*conf
                
                if self.level >=3 :
                    cost5 = F.interpolate((cost5).unsqueeze(1), [focal_dist.shape[1], h, w], mode='trilinear').squeeze(1)
                    cost_stacked.append(cost5)
                    if(self.fuse==0 or self.fuse==2):
                        if(self.disp_mode==0):
                            fdepth5,std5=self.disp_reg(F.softmax(cost5,1),focal_dist, uncertainty=True)
                            stds.append(std5)
                            #print('fdepth5:'+str(fdepth5.shape))
                        elif(self.disp_mode==1):
                            fdepth5=self.dispNet(cost5,foc_ar)
                            #print('fdepth5_net:'+str(fdepth5.shape))
                        fstacked.append(fdepth5)
                    if(self.fuse==2):
                        pred5=(fdepth5+ddepth)*0.5
                        fdstacked.append(pred5)

                    if self.level >=4 :
                        cost6 = F.interpolate((cost6).unsqueeze(1), [focal_dist.shape[1], h, w], mode='trilinear').squeeze(1)
                        cost_stacked.append(cost6)
                        if(self.fuse==0 or self.fuse==2):
                            if(self.disp_mode==0):
                                fdepth6,std6=self.disp_reg(F.softmax(cost6,1),focal_dist, uncertainty=True)
                                stds.append(std6)
                                #print('fdepth6:'+str(fdepth6.shape))
                            elif(self.disp_mode==1):
                                fdepth6=self.dispNet(cost6,foc_ar)
                                #print('fdepth6_net:'+str(fdepth6.shape))
                            fstacked.append(fdepth6)
                        if(self.fuse==2):
                            pred6=(fdepth6+ddepth)*0.5
                            fdstacked.append(pred6)
                        
            return fstacked,ddepth,fdstacked,stds,cost_stacked
        #if evaluating the model
        else:
            if(self.fuse==0 or self.fuse==2):
                std3=torch.squeeze(std3)
            return fdepth3,ddepth,fddepth3,std3,cost3

#model=DFFNet(clean=0,level=4,use_diff=0,fuse=1,disp_mode=0)
#stack=torch.rand(4,5,3,224,224)
#focal_dist=torch.rand(1,5)
#out=model(stack,focal_dist)
