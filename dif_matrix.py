#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
	Created on Apr 2019
	
	@author: ZYW @ BNU
	'''
#creat a matrix of derivation of 5 points

import numpy as np
import matplotlib.pyplot as plt

lens_sides = 256
#field degree (have to be a squared field)
field_deg = 20.0
#lensing amplitude magnification factor
#factor of scals
ipn = 5
#number of interplate points
h_0 = field_deg/lens_sides
#step-size of derivative

def g(k):
	if k in range(lens_sides):
		return k
	elif k in range(lens_sides,lens_sides+ipn):
		return k-lens_sides
	elif k in range(-1*ipn,0):
		return k+lens_sides
#establish a periodic fuction of map cyclic boundary

di_e = np.zeros(shape=(1,ipn),dtype='float')
di = np.zeros(shape=(lens_sides,lens_sides),dtype='float')

di_e = [-1.,-8.,0.,8.,1.]
#generating vector of derivation matrix

for i in range(lens_sides):
	for j in range(ipn):
		di[i,g(i-2+j)] = di_e[j]
dif=di/12.0/h_0
#Assign value to derivation matrix

np.savetxt('diff_matirx.dat', dif)
#saving derivation matrix
exit()