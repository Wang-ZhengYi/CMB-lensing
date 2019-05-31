#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Apr 2019
    
@author: BH & WZY @ BNU
'''

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

#number of pixel
lens_sides = 256
half_lens_side = 128
p = 2.0
magnif = 3e10
field_deg=20.0
h_0 = field_deg/lens_sides
t=20

fore_lens_map_data = np.zeros(shape=(lens_sides,lens_sides),dtype='float')
lens_map_data = np.zeros(shape=(lens_sides,lens_sides),dtype='float')

for i in range(lens_sides):
	for j in range(lens_sides):
			lens_map_data[i,j] = magnif*1/(sqrt(((i-half_lens_side-0.5)*h_0)**2+((j-half_lens_side-0.5)*h_0)**2)+t)**p
#Assign value to matrix lens_map_data


#saving lensing potential data
np.savetxt('lens_data.dat', lens_map_data)

#plot lensing potential
lens_data_path = './lens_data.dat'
lens_data = np.loadtxt(lens_data_path)
plt.imshow(lens_data,cmap='binary')
#plt.show()
plt.axis('on')
plt.colorbar()
plt.savefig('lens_map.png')

exit()

