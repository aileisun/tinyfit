""" Contain small functions for imgfitter """

import numpy as np
import scipy.ndimage as sn
import photutils as pu
from astropy.io import fits
from scipy import optimize
from scipy.interpolate import UnivariateSpline
import astropy.table as at


def _find_a_root(x, y ): 
	spl = UnivariateSpline(x, y, k=1, s=0)
	root = optimize.newton(spl, 2.)
	return root


def petrosian_radiu(data, r_step=5., petrosian_ratio=0.2):
	""" measure petrosian radiu in units of pixel from data. 
	
	Note: 
		Centroid position is determined from data

	Args: 
		data (2d np array)
		r_step=5. (float): aperture radius steps in units of pixel
		petrosian_ratio=0.2: the surface brightness at r_petrosian is petrosian_ratio times the surface brightness within r_petrosian. 
	"""
	ny, nx = data.shape
	xc, yc = pu.centroid_1dg(data)
	rs = np.arange(r_step, ny//2, r_step)

	apts = [pu.CircularAperture(positions=(xc, yc), r=r) for r in rs]

	tab = at.Table(pu.aperture_photometry(data, apts))

	if len(rs) == 1:
		cols = ['aperture_sum']
	else: 
		cols = ['aperture_sum_'+str(i) for i in range(len(rs))]

	aperture_sum = np.array(list(tab[cols][0]))
	annulus_sum = np.zeros_like(rs)
	annulus_sum[0] = aperture_sum[0]
	for i in range(1, len(rs)):
		annulus_sum[i] = aperture_sum[i] - aperture_sum[i-1]

	aperture_areas = np.pi*rs**2
	annulus_areas = np.append(aperture_areas[0], np.diff(aperture_areas))

	annulus_avg = annulus_sum/annulus_areas
	aperture_avg = aperture_sum/aperture_areas

	diff = annulus_avg - aperture_avg*petrosian_ratio

	r_petrosian = _find_a_root(x=rs, y=diff)

	return r_petrosian


def get_cutout_xy_range(xc, yc, nx, ny):
	""" get cutout range x0, x1, y0, y1 given xc, yc, nx, ny """
	xc, yc = int(xc), int(yc)

	if nx%2 == 0:
		x0, x1 = xc-nx//2, xc+nx//2
	elif nx%2 == 1:
		x0, x1 = xc-nx//2, xc+nx//2 + 1

	if ny%2 == 0:
		y0, y1 = yc-ny//2, yc+ny//2
	elif ny%2 == 1:
		y0, y1 = yc-ny//2, yc+ny//2 + 1

	return x0, x1, y0, y1


def shift(data, dx, dy, subsample=1):
	"""shift data by [dy, dx] pixels (original pix, not data pix) via linear interpolation. 

	Args: 
		data (2d np array)
		dx (float): amount shifted in x in units of original pixel size
		dy (float): amount shifted in y in units of original pixel size
		subsample (int): subsampling factor


	Note:
		Original pixel size is the subsample factor times the input data pixelsize. dx, dy is expressed in original pixel size not the input data pixel size. 
	"""
	# sanity check
	if type(subsample) is not int:
		raise Exception("subsample factor is not integer")

	data_shift = sn.interpolation.shift(data, shift=[dy*subsample, dx*subsample], order=1, mode='constant', cval=0.)

	return data_shift


def resample(data, nx, ny, subsample=1):
	"""
	Resample data that is subsampled by a integer factor (subsample) back to original sample and resize it to shape nx, ny.

	Args: 
		data (2d np array)
		nx (int): dimension of the output array
		ny (int): dimension of the output array
		subsample (int): subsampling factor

	Return:
		data_result (2d np array): of size [ny, nx]

	Note:
		Data is regridded to the desired pixsize by integrating n*n pixels into one pixel, where n is the subsampling factor. In practice, it is done by convoluting the model with a 2d top-hat function of size n*n and resampling every n pixels. 
	"""
	# sanity check
	if type(subsample) is not int:
		raise Exception("subsample factor is not integer")
	if (type(nx) is not int) or (type(ny) is not int):
		raise Exception("array dimension is not integer")

	# setting
	yc, xc = ny//2, nx//2
	ny_subsamp, nx_subsamp = data.shape
	yc_subsamp, xc_subsamp = np.array(data.shape)//2

	# convolve
	kernel = np.ones([subsample, subsample])
	data_conv = sn.convolve(data, kernel, mode='constant', cval=0.)

	# resample
	x = np.arange(xc_subsamp%subsample, nx_subsamp, step=subsample)
	y = np.arange(yc_subsamp%subsample, ny_subsamp, step=subsample)
	xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')
	data_resamp = data_conv[yv, xv]

	# padding
	ny_resamp, nx_resamp = data_resamp.shape
	yc_resamp, xc_resamp = np.array(data_resamp.shape)//2
	dx = xc - xc_resamp
	dy = yc - yc_resamp

	data_result = np.pad(data_resamp, [[dy, ny-ny_resamp-dy], [dx, nx-nx_resamp-dx]], mode='constant', constant_values=0.)

	return data_result