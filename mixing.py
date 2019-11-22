#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
	Created on Sep 2018
	
	@author: BH & ZYW @ BNU
	'''

#from PIL import Image
import numpy as np
#from scipy.interpolate import Rbf 
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import interpolate
#from pylab import *
#import astropy.units as u

##-----------------parameter setting--------------------##
magnif_pixel = 4
#line resolution for lensing potential map is N/magnif_pixel when line resolution for cmb map is N
cmb_map_sides = 1024
#number of pixel
lens_sides = 256
#field degree (have to be a squared field)
field_deg = 20.0 #u.deg
#lensing amplitude magnification factor
magnif = 1.5e-5
#factor of scals
#step-size of derivative

h_0 = field_deg/lens_sides

lens_map_data_path = './lens_data.dat'
lens_map_data = np.loadtxt(lens_map_data_path)
lens_map_data = magnif*lens_map_data
#load lens map data as a matrix

diff_path = './diff_matirx.dat'
diff_matrix = np.loadtxt(diff_path)
diff_matrixT = np.transpose(diff_matrix)
#transform gradient calculations to matrix calculations: psi_x = LD,psi_y = D(T)L

cmb_map_data_path = './map_data.dat'
cmb_map_data = np.loadtxt(cmb_map_data_path)
#load cmb map data as a matrix

cmb_map_dataT=np.transpose(cmb_map_data)
size0 = np.shape(cmb_map_data)
xn , yn = size0
xx = np.arange(xn)
yy = np.arange(yn)
zz = cmb_map_dataT
interped_cmb_map_data = interpolate.interp2d(xx, yy, zz, kind='cubic')
#import data and interpolate a 2d consecutive cmb map,here transpose cmb_map_data due to python interpolation convention

lensed_cmb_map_data = np.zeros(shape=(cmb_map_sides,cmb_map_sides),dtype='float')
cmb_alpha_x = np.zeros(shape=(cmb_map_sides,cmb_map_sides),dtype='float')
cmb_alpha_y = np.zeros(shape=(cmb_map_sides,cmb_map_sides),dtype='float')
alpha_x = np.zeros(shape=(lens_sides,lens_sides),dtype='float')
alpha_y = np.zeros(shape=(lens_sides,lens_sides),dtype='float')

#lensed cmb map part:
alpha_x = np.dot(lens_map_data,diff_matrix)
#compute the deflection field: x-component
alpha_y = np.dot(diff_matrixT,lens_map_data)
#compute the deflection field: y-component

for p in range(lens_sides):
	for q in range(lens_sides):
		for i in range(magnif_pixel):
			for j in range(magnif_pixel):
				cmb_alpha_x[magnif_pixel*p+i,magnif_pixel*q+j]=alpha_x[p,q]
				cmb_alpha_y[magnif_pixel*p+i,magnif_pixel*q+j]=alpha_y[p,q]
#creat a map of deflection angle map the same resolution as cmb map

for p in range(cmb_map_sides):
	for q in range(cmb_map_sides):
		x_1 = q-cmb_alpha_x[p,q]
		y_1 = p-cmb_alpha_y[p,q]
		lensed_cmb_map_data[q,p] = interped_cmb_map_data(x_1,y_1)
#Assign value to matrix lensed_cmb_map_data
#moving pixels

plt.imshow(lensed_cmb_map_data,cmap='bwr',vmin=-1e-5,vmax=1e-5)
plt.axis('on')
plt.colorbar()
#plt.show()
plt.savefig('lensed_cmb_map.png')
#print lensed cmb map

exit()
