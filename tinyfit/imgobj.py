# -*- coding: utf-8 -*-

"""
class imgobj that contains basic info of an image such as pixsize. 
"""
import numpy as np
from astropy.io import fits


class imgobj(object):
	"""imgobj contains image information -- data, pixsize, nx, ny, xc, yc. 

	Example:
		>>> img = imgobj(filename='my_image.fits', pixsize=0.13)

		>>> fn = 'my_image.fits'
		>>> data = fits.getdata(fn)
		>>> img = imgobj(data=data, pixsize=0.13)
		>>> img.writeto('anotherfile.fits')

	Note:
		Image center is set to [ny//2, nx//2]. 
		Index convension is [y, x]

	Attributes:
		data (2d np array)
		pixsize (float)
		nx (int): image dimension
		ny (int): image dimension
		xc (int): image center
		yc (int): image center
	"""
	def __init__(self, **kwargs):
		"""
		Args:
			filename (str) : File path to fits file of the image to be fitted to. 
				-- or alternatively --
				fn (str) : File path to fits file of the image to be fitted to. 
				data (:obj:'np.array'): 2d image array of floats. 
			pixsize (float, optional) : pixelsize in arcsec. 
		"""
		if ('filename' in kwargs) or ('fn' in kwargs):
			if 'filename' in kwargs: 
				self.filename = kwargs.pop('filename', None)
			else: 
				self.filename = kwargs.pop('fn', None)
			self.data = fits.getdata(self.filename)
			self.header = fits.getheader(self.filename)
			if 'PIXSCALE' in self.header:
				pixsize = self.header['PIXSCALE']
			else:
				pixsize = None
		elif 'data' in kwargs:
			self.data = kwargs.pop('data', None)

			if type(self.data) is not np.ndarray:
				raise Exception("Input data is not numpy array. ")

			pixsize = None
		else: 
			raise Exception('imgobj requires argument of either filename or data')

		self.pixsize = kwargs.pop('pixsize', pixsize)
		self.ny, self.nx = self.data.shape
		self.yc, self.xc = self.ny//2, self.nx//2


	def writeto(self, filename, overwrite=True):
		""" write image to fits file at filename 
		Args:
			filename (str): filename to save fits file to
			overwrite=True (bool): whether to overwrite file. 
		"""
		hdu = fits.PrimaryHDU(self.data)
		if self.pixsize is not None:
			hdu.header['PIXSCALE'] = self.pixsize

		hdu.writeto(filename, overwrite=overwrite)