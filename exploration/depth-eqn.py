#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 09:27:07 2022

@author: sleekeagle
"""
import matplotlib.pyplot as plt
import numpy as np


s1 = np.linspace(0.1,2.0,100)
f=2.9 * 1e-3
N=1.0
s2=0.05

# setting the axes at the centre
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.xlim(0,2.0)
#plt.plot(s1,c)

#plt.ylim(0,5)
plt.xlim(0,2.0)
lines=[]
s2_vals=[0.05,0.1,0.5,1.0,2.0]
for s2 in s2_vals:
    c=abs(s2-s1)/(s2*N*(s1-f))*np.square(f)
    l,=ax.plot(s1,c)
    lines.append(l)
ax.legend(lines,[str(v) for v in s2_vals],loc='upper right')
ax.set_xlabel("S1")
ax.set_ylabel("CoC radius")
ax.set_title('CoC radius vs S1 for different S2 values')
fig.savefig('/home/sleekeagle/vuzix/depth-from-Focus-Defocus/exploration/Coc-vs-s1.png',bbox_inches='tight',dpi=1200)

'''
plot the abs value of derivative of CoC wrt S2
'''

# setting the axes at the centre
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.xlim(0,2.0)
s2=2.0
#find the absolute value of derivative of c wrt s2
dcs2 = s1*np.square(f)/(np.square(s2)*N*(s1-f))
l,=ax.plot(s1,dcs2)
ax.set_xlabel("S1")
ax.set_ylabel("d(CoC radius)/ds2")
ax.set_title('d(CoC radius)/ds2 vs S1 for s2=2.0')
fig.savefig('/home/sleekeagle/vuzix/depth-from-Focus-Defocus/exploration/d-Coc-vs-s1 s2=2.0.png',bbox_inches='tight',dpi=1200)

'''
plot the average derivative value of CoC 
'''
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.xlim(0,2.1)
s2_vals=[0.05,0.1,0.5,1.0,2.0]
means=[]
for s2 in s2_vals:
   dcs2 = s1*np.square(f)/(np.square(s2)*N*(s1-f))
   means.append(np.mean(dcs2))

plt.scatter(s2_vals,means)
plt.title("average derivative of CoC wrt s2 over s1=[0.1,2]")
plt.xlabel("s2")
plt.ylabel("average derivative")
fig.savefig('/home/sleekeagle/vuzix/depth-from-Focus-Defocus/exploration/avg_der.png',bbox_inches='tight',dpi=1200)






