""" Contain small functions for imgfitter """

import numpy as np
import scipy.ndimage as sn


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