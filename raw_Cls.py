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

#Units
#import astropy.units as u
#from astropy.units import deg

#file name
unlensed_spectra_filename = './lcdm_scalCls.dat'

##-------parameter setting-----------##

#number of pixel
cmb_map_sides = 1024
#field degree (have to be a squared field)
field_deg = 20
#comoving distance
chi = 14000.0 #in the unit of [Mpc]


##------------------##

#reading the spectrum data
unlensed_spectra = np.loadtxt(unlensed_spectra_filename)
#print(unlensed_spectra.shape[0])
ell = np.zeros(unlensed_spectra.shape[0])
TT_unlensed = np.zeros(unlensed_spectra.shape[0])
EE_unlensed = np.zeros(unlensed_spectra.shape[0])

ell[0:len(ell)] = unlensed_spectra[:,0]

TT_unlensed[0:len(ell)] = unlensed_spectra[:,1]#/unlensed_spectra[:,0]/(unlensed_spectra[:,0]+1)*2.0*np.pi

EE_unlensed[0:len(ell)] = unlensed_spectra[:,2]#/unlensed_spectra[:,0]/(unlensed_spectra[:,0]+1)*2.0*np.pi

l = np.arange(2,6000,0.01)

TT_interp = interpolate.interp1d(ell,TT_unlensed,bounds_error=False,kind='linear',fill_value=0.0)
TTl = TT_interp(l)
EE_interp = interpolate.interp1d(ell,EE_unlensed,bounds_error=False,kind='linear',fill_value=0.0)
EEl = EE_interp(l)

fig = plt.figure()

ax1 = fig.add_subplot(211)
ax1.plot(l,TTl,'b-')
plt.xscale('log')


ax2 = fig.add_subplot(212)
ax2.plot(l,EEl,'r-')
plt.xscale('log')

plt.show()
exit()