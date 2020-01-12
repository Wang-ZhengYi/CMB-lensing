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
	def __init__(self, arg):
		super(lens, self).__init__()
		self.arg = arg

	def der1(maps,field):
		Rm = diffmat(maps.shape[0])/field[0]
		Wm = diffmat(maps.shape[1])/field[1]
		X = np.dot(maps,Rm)
		Y = -1*np.dot(Wm,maps)
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
	a = lens.flat_lens(cmb_0,phi_0,fieldeg)
	plt.imshow(a,
		cmap='bwr',
		 # vmin=0,vmax=8000
		)
	plt.axis('on')
	plt.colorbar()
	
	# plt.savefig('cmbl1.png',dpi=600)
	plt.show()
	exit()