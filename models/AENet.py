# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as  F

# static architecture
class AENet(nn.Module):

    def __init__(self,out_dim, num_filter,nstacks, n_blocks=3, flag_step2=False):
        super(AENet, self).__init__()
        
        self.out_dim = out_dim
        self.num_filter = num_filter
        self.n_blocks = n_blocks
        act_fnc = nn.LeakyReLU(0.2, inplace=True)


        self.conv_down2_0 = self.convsblocks(2, self.num_filter * 1, act_fnc)
        self.pool2_0 = self.poolblock()


        for i in range(self.n_blocks):
            self.add_module('conv_down2_' + str(i + 1), self.convsblocks(self.num_filter * (2 ** i) * 2, self.num_filter * (2 ** i) * 2, act_fnc))
            self.add_module('pool2_' + str(i + 1), self.poolblock())

        self.bridge2 = self.convblock(256, self.num_filter * 16, act_fnc)

        for i in range(self.n_blocks + 1):
            self.add_module('conv_up2_' + str(i + 1),
                            self.upconvblock(int(self.num_filter * (2 ** (3 - i)) * 2), int(self.num_filter * (2 ** (3 - i))), act_fnc))
            self.add_module('conv_joint2_' + str(i + 1),
                            self.convblock(int(self.num_filter * (2 ** (3 - i)) * 3), int(self.num_filter * (2 ** (3 - i))), act_fnc))

        self.conv_end2 = self.convblock(self.num_filter * 1, self.num_filter * 1, act_fnc)

        self.conv_out2 = nn.Sequential(
            nn.Conv2d(self.num_filter, self.out_dim, kernel_size=3, stride=1, padding=1),
        )


        
    def convsblocks(self, in_ch,out_ch,act_fn):
        block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            act_fn,
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),            
            act_fn,
        )
        return block
    
    def convblock(self, in_ch,out_ch,act_fn):
        block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            act_fn,
        )
        return block
    
    def upconvblock(self,in_ch,out_ch,act_fn):
        block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            act_fn,
        )
        return block
    
    def poolblock(self):
        pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        return pool


    def forward(self, x,down1,k=8,x2=0):
        down2 = []
        pool_temp = []
        for j in range(self.n_blocks + 1):
            down_temp = []
            #print("@@@@@")
            for i in range(k):
                #print('******')
                if j > 0:
                    joint_pool = torch.cat([pool_temp[0], pool_max[0]], dim=1)
                    pool_temp.pop(0)
                else:
                    #print(x2[:, 1 * i:1 * (i + 1), :, :].shape)
                    joint_pool = torch.cat([x[:, 1 * i:1 * (i + 1), :, :],x2[:, 1 * i:1 * (i + 1), :, :]], dim=1)
                
                #print(joint_pool.shape)
                conv = self.__getattr__('conv_down2_' + str(j + 0))(joint_pool)
                down_temp.append(conv)
                pool = self.__getattr__('pool2_' + str(j + 0))(conv)
                #print('pool : '+str(pool.shape))
                pool_temp.append(pool)
                pool = torch.unsqueeze(pool, 2)
                #print('pool : '+str(pool.shape))
                if i == 0:
                    pool_all = pool
                else:
                   pool_all = torch.cat([pool_all, pool], dim=2)
    
            pool_max = torch.max(pool_all, dim=2)
            down2.append(down_temp)

        #print('len pool temp='+str(len(pool_temp)))
        bridge = []
        for i in range(k):
            join_pool = torch.cat([pool_temp[i], pool_max[0]], dim=1)
            bridge.append(self.bridge2(join_pool))


        #print('bridge shapes:')
        #for i in range(len(bridge)):
        #    print(bridge[i].shape)
            
        #print('seconds step: ')
        up_temp = []
        for j in range(self.n_blocks + 2):
            #print('pp@@@')
            for i in range(k):
                #print('****')
                if j > 0:
                    joint_unpool = torch.cat([up_temp[0], unpool_max[0], F.interpolate(down1[j-1][:,i,:,:],[up_temp[0].shape[-1],up_temp[0].shape[-1]],mode='bilinear')], dim=1)
                    up_temp.pop(0)
                    joint = self.__getattr__('conv_joint2_' + str(j + 0))(joint_unpool)
                else:
                    joint = bridge[i]

                if j < self.n_blocks + 1:
                    unpool = self.__getattr__('conv_up2_' + str(j + 1))(joint)
                    #print(unpool.shape)
                    up_temp.append(unpool)
                    unpool = torch.unsqueeze(unpool, 2)

                    if i == 0:
                        unpool_all = unpool
                    else:
                        unpool_all = torch.cat([unpool_all, unpool], dim=2)
            unpool_max = torch.max(unpool_all, dim=2)

        end2 = self.conv_end2(unpool_max[0])
        out_step2 = self.conv_out2(end2)
        
        return out_step2
      
        
        
#model=AENet(out_dim=1,nstacks=6,num_filter=16)
#model
#bs=2
#down1=[torch.rand((bs,6,128,7,7)),torch.rand((bs,6,64,14,14)),torch.rand((bs,6,32,28,28)),torch.rand((bs,6,16,56,56))]
#img=torch.rand((bs,6,224,224))
#fd=torch.rand(bs,6,224,224)
#out=model(img,down1,6,fd)
#out.shape
