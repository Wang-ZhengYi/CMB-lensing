#!/usr/bin/python3
#-*- coding:utf-8 -*-
'''
created on Nov.,2019

@Author:Zhengyi Wang@BNU
'''


import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from scipy import interpolate
import camb
import math
import gravipy
import astropy
 

def coord(xsides,ysides):
	x = np.arange(xsides)
	y = np.arange(ysides)
	coordm = np.zeros(shape=(x.shape[0],y.shape[0]),dtype='complex')
	coordm = x[np.newaxis,:] + y[:,np.newaxis]*1.0j
	return coordm,x,y


def border(x,N):
	# x = x.astype(np.int)
	more = (x>=N)
	less = (x<0)
	return x-N*more+N*less

def diffmat(N):
	di_e = np.array([-1.,-8.,0.,8.,1.])/12.0
	M = np.zeros(shape=(N,N),dtype='float')

	for j in range(len(di_e)):
		for i in range(N):
			M[i,border(i-len(di_e)//2+j,N)] = di_e[j]*N

	return M

		

class lens(object):
	"""docstring for lens"""
	'''
	this class is a naive approach to calculate the gravitational lensing of CMB
	'''
	def __init__(self, arg):
		super(lens, self).__init__()
		self.arg = arg

	def der1(maps,field):
		Rm = diffmat(maps.shape[0])/field[0]
		Wm = diffmat(maps.shape[1])/field[1]
		X = np.dot(maps,Rm)
		Y = -1*np.dot(Wm,maps)
		'''
		the defult method of derivative in python cannot make users satisfied in computing time.
		Here is a linear approximation of derivatives of a 2D map in matrices calculations,which increases the burden of memory,but speed up the operations
		'''
		return X,Y

	def der2(maps,field):
		x,y = lens.der1(maps,field)
		xx,xy = lens.der1(x,field)
		yx,yy = lens.der1(y,field)
		return xx,yy,xy,yx
 
	def flatlens(cmb,phi,field):
		#field is the field of the map whose unit is deg
		cmb_x,cmb_y = lens.der1(cmb,field)
		phi_x,phi_y = lens.der1(phi,field)
		cmb_xx,cmb_yy,cmb_xy,cmb_yx = lens.der2(cmb,field)
		cmbl = cmb + (cmb_x*phi_x + cmb_y*phi_y) + 0.5*(cmb_xx*phi_x**2+cmb_yy*phi_y**2+cmb_xy*phi_y*phi_x+cmb_yx*phi_y*phi_x)
		return cmbl

	def flat_lens(cmb,phi,field):
		coordm,xx,yy = coord(cmb.shape[0],cmb.shape[1])
		zz = cmb
		phi_x,phi_y = lens.der1(phi,field)
		interped_cmb = interpolate.interp2d(xx, yy, zz, kind='cubic')
		coordm = coordm -phi_x - phi_y*1.0j
		x,y = np.meshgrid(xx,yy)
		for i in range(cmb.shape[0]):
			for j in range(cmb.shape[1]):				
				cmb[i,j] = interped_cmb(coordm[i,j].imag,coordm[i,j].real)
		return cmb 

	def flat_lens_sb(cmb,phi,field):
		cmbk = np.fft.fft2(cmb)/cmb.shape[0]
		return cmbk


	def curvelens(cmb,phi,nside):
		alm_cmb = hp.map2alm(cmb)
		alm_phi = hp.map2alm(phi)
		grad_alm_cmb = hp.alm2map_der1(alm_cmb,nside)
		grad_alm_phi = hp.alm2map_der1(alm_phi,nside)
		lensedmap = cmb + grad_alm_phi*grad_alm_cmb
		return lensedmap

	# def curve_lens(cmb,phi,nside):


class flat(object):
	"""docstring for flatcmb"""
	def __init__(self, arg):
		super(flatcmb, self).__init__()
		self.arg = arg

	def k2map(c_ell,nside,field_deg,l_min,l_max):
		ell = len(c_ell)
		l_min = 360.0/field_deg #ell minimum ell=2pi/theta
		l_max = nside * l_min
		ly = np.fft.fftfreq(nside)*l_max #* l_max
		lx = np.fft.rfftfreq(nside)*l_max
		l = np.sqrt(lx[np.newaxis,:]**2 + ly[:,np.newaxis]**2)
		power_interp = interpolate.interp1d(ell,TT_unlensed,bounds_error=False,kind='linear',fill_value=0.0)
		Pl = power_interp(l)
		# np.random.seed(1)
		real_part = np.sqrt(0.5*Pl) #* np.random.normal(loc=0.0,scale=1.0,size=l.shape)
		imaginary_part = np.sqrt(0.5*Pl) #* np.random.normal(loc=0.0,scale=1.0,size=l.shape)
		ft_map = real_part + imaginary_part*1.0j
		noise_map = np.fft.irfft2(ft_map)
		return noise_map

	def map2k(maps,field_deg,k_max):
		field_deg = np.float64(field_deg)
		# k_max = k_max.astype(np.int)
		nside = maps.shape[0]
		# print(nside)
		ft_map = np.fft.rfft2(maps)*nside
		P = abs(ft_map)**2
		
		l_min = 360.0/field_deg #ell minimum ell=2pi/theta
		l_max = nside * l_min
		ly = np.fft.fftfreq(nside)*l_max #* l_max
		lx = np.fft.rfftfreq(nside)*l_max
		l = np.sqrt(lx[np.newaxis,:]**2 + ly[:,np.newaxis]**2)
		p_l = 1.0j*P + l
		pl = np.zeros(nside*(nside//2+1),dtype='complex')
		p_l = np.sort_complex(p_l)

		p_l = p_l.reshape((1,len(pl)))
		pl[:] = p_l[0,:]
		# print(pl.shape)
		p_ll = interpolate.interp1d(pl.real,pl.imag,bounds_error=False,kind='linear',fill_value=0.0)
		ell = np.arange(k_max)
		C_l = p_ll(ell)
		return ell,C_l
	
	def der1(maps,field):
		Rm = diffmat(maps.shape[0])/field[0]
		Wm = diffmat(maps.shape[1])/field[1]
		X = np.dot(maps,Rm)
		Y = -1*np.dot(Wm,maps)
		'''
		the defult method of derivative in python cannot make users satisfied in computing time.
		Here is a linear approximation of derivatives of a 2D map in matrices calculations,which increases the burden of memory,but speed up the operations
		'''
		return X,Y

	def der2(maps,field):
		x,y = lens.der1(maps,field)
		xx,xy = lens.der1(x,field)
		yx,yy = lens.der1(y,field)
		return xx,yy,xy,yx
 
	def flatlensX(cmb,phi,field):
		#field is the field of the map whose unit is deg
		cmb_x,cmb_y = lens.der1(cmb,field)
		phi_x,phi_y = lens.der1(phi,field)
		cmb_xx,cmb_yy,cmb_xy,cmb_yx = lens.der2(cmb,field)
		cmbl = cmb + (cmb_x*phi_x + cmb_y*phi_y) + 0.5*(cmb_xx*phi_x**2+cmb_yy*phi_y**2+cmb_xy*phi_y*phi_x+cmb_yx*phi_y*phi_x)
		return cmbl

	# def flatlensK():


	def flat_lens(cmb,phi,field):
		coordm,xx,yy = coord(cmb.shape[0],cmb.shape[1])
		zz = cmb
		phi_x,phi_y = lens.der1(phi,field)
		interped_cmb = interpolate.interp2d(xx, yy, zz, kind='cubic')
		coordm = coordm - phi_x - phi_y*1.0j
		x,y = np.meshgrid(xx,yy)
		for i in range(cmb.shape[0]):
			for j in range(cmb.shape[1]):				
				cmb[i,j] = interped_cmb(coordm[i,j].imag,coordm[i,j].real)
		return cmb		

class curve(object):
	"""docstring for curve"""
	def __init__(self, arg):
		super(curve, self).__init__()
		self.arg = arg

	def curvelens(cmb,phi,nside):
		alm_cmb = hp.map2alm(cmb)
		alm_phi = hp.map2alm(phi)
		grad_alm_cmb = hp.alm2map_der1(alm_cmb,nside)
		grad_alm_phi = hp.alm2map_der1(alm_phi,nside)
		lensedmap = cmb + grad_alm_phi*grad_alm_cmb
		return lensedmap

		
unlensed_spectra_filename = './lcdm_scalCls.dat'

##-------parameter setting-----------##

#number of pixel
cmb_map_sides = 1024
#field degree (have to be a squared field)
field_deg = 20.
#comoving distance
chi = 14000.0 #in the unit of [Mpc]


##------------------##

#reading the spectrum data
unlensed_spectra = np.loadtxt(unlensed_spectra_filename)
#print(unlensed_spectra.shape[0])
el = np.zeros(unlensed_spectra.shape[0])
TT_unlensed = np.zeros(unlensed_spectra.shape[0])
el[0:len(el)] = unlensed_spectra[:,0]
TT_unlensed[0:len(el)] = unlensed_spectra[:,1]#/unlensed_spectra[:,0]/(unlensed_spectra[:,0]+1)*2.0*np.pi

magnif = 1e-5

lens_map_data_path = './lens_data.dat'
lens_map_data = np.loadtxt(lens_map_data_path)
phi_0 = magnif*lens_map_data
#load lens map data as a matrix

cmb_map_data_path = './map_data.dat'
cmb_0 = np.loadtxt(cmb_map_data_path)
#load cmb map data as a matrix

fieldeg = np.array([20.,20.])

if __name__ == '__main__':
	# lx,ly = lens.der1(phi_0,fieldeg)
	elll,C_l = flat.map2k(cmb_0,20.,1000)
	plt.plot(elll[50:999],C_l[50:999],'b-')
	plt.plot(el[5:999],TT_unlensed[5:999],'r-')
	# plt.set_xlim(2,1000)
	plt.axis('on')
	# plt.colorbar()
	
	# plt.savefig('cmbl1.png',dpi=600)
	
	plt.imshow()

	plt.show()
	exit()