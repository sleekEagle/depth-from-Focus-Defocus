import numpy as np
import matplotlib.pyplot as plt

N=2
f=9.5*1e-3

s1range=np.arange(0.1,2,0.1)
s2=0.5
s1=0.5
s2range=np.arange(0.1,2,0.1)

#calculate blur*N/f^2 = |1-s1/s2|/(s1-f)

x=abs(1-s1range/s2)/(s1range-f)
plt.plot(x)
plt.show()

x=abs(1-s1/s2range)
plt.plot(x)
plt.show()


#error of the |1-s1/s2|
s=0.5
sest=np.arange(0.1,2,0.1)
sest=0.3
s1range=np.arange(0.1,2,0.1)+np.random.rand()*0.99
s1=0.4

import math
def getblurish(s1,s2):
    blurish=abs(1-s1/s2)
    return blurish
def get_derr(s1,sest):
    derr=2*s1/(sest**2) -2*(1-s1/s -s1/sest+s1**2/(s*sest))/(abs(1-s1/s -s1/sest+s1**2/(s*sest))) * (s1/sest**2 - s1**2/(s*sest**2))
    return derr

s1=np.arange(0.01,2,0.3)
s2=1.0

blurish=getblurish(s1,s2)
#add noise
blurish=blurish+np.random.rand(7)*0.1
bd=np.diff(blurish)/np.diff(s1)
plt.plot(s1,blurish,marker='o')
plt.plot(s1[0:-1],bd,marker='o')
plt.show()
blurish=blurish[8:15]
s1=s1[8:15]

import torch
s1=torch.tensor(s1)
blurish=torch.from_numpy(blurish)
bdiff=torch.diff(blurish)
sdiff=torch.diff(s1)
#are there zero crossings?


import torchpwl
pwl = torchpwl.PWL(num_channels=1, num_breakpoints=1)
opt = torch.optim.Adam(params=pwl.parameters(), lr=0.0001)
s1_=np.expand_dims(s1,axis=1)
#s1_=np.repeat(s1_,10,axis=1)
s1_=torch.from_numpy(s1_)
blurish_=np.expand_dims(blurish,axis=1)
#blurish_=np.repeat(blurish_,10,axis=1)
blurish_=torch.from_numpy(blurish_)

losses=[]
for i in range(1000):
    pred = pwl(s1_)
    loss=torch.mean((pred-blurish_)**2)
    losses.append(loss.item())
    opt.zero_grad()
    loss.backward()
    opt.step()


pwl.get_slopes()





x = torch.Tensor(11, 5).normal_()


import pwlf

s1_=np.expand_dims(s1,axis=1)
s1_=np.expand_dims(s1_,axis=2)
s1_=np.repeat(s1_,10,axis=1)
s1_=np.repeat(s1_,10,axis=2)

blurish_=np.expand_dims(blurish,axis=1)
blurish_=np.expand_dims(blurish_,axis=2)
blurish_=np.repeat(blurish_,224,axis=1)
blurish_=np.repeat(blurish_,224,axis=2)

def get_linfit(x):
    myPWLF = pwlf.PiecewiseLinFit(s1,x)
    z = myPWLF.fit(2)
    return np.array([3])
slopes = myPWLF.calc_slopes()

np.apply_along_axis(get_linfit,0,blurish_).shape


def get_val(x,y):
    print(type(y))
    print(type(x))
    print(y)
    print(x.shape)
    return np.random.rand(2,3,4)

a = np.arange(24).reshape(2,3,4)
np.apply_over_axes(get_val, a, [1,2]).shape


def gets2(tensor):
    std=torch.std(tensor,axis=1,keepdim=True).repeat_interleave(tensor.shape[1],1)
    mean=torch.mean(tensor,axis=1,keepdim=True).repeat_interleave(tensor.shape[1],1)
    far=torch.abs((out-mean)/std)
    outlier=far<1.6
    clean=torch.where(outlier,out,mean)
    s2=1/torch.mean(clean,axis=1,keepdim=True)
    return s2

out=torch.rand((4,5,224,224))
gets2(out).shape


torch.mean(torch.tensor([0.5384, 0.6065, 0.4123, 0.0921, 0.3202, 0.3226]))

out=torch.tensor([0.4,0.5,0.33,0.9,0.54,0.55])
mean=torch.mean(out)
far=torch.abs((out-mean)/std)
outlier=far<1.6
s2=1/torch.mean(out[outlier])














