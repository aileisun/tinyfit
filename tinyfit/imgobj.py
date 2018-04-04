# -*- coding: utf-8 -*-

"""
class imgobj that contains basic info of an image such as pixsize. 
"""

from astropy.io import fits


class imgobj(object):
	def __init__(self, **kwargs):
		"""
		imgobj (class)

		Contains image information. The index convension is data[y, x]. 

		Note
		----
		Image center is set to [ny//2, nx//2]

		Parameters
		----------
		filename (str) : File path to fits file of the image to be fitted to. 
			-- or alternatively --
			fn (str) : File path to fits file of the image to be fitted to. 
			data (:obj:'np.array'): 2d image array of floats. 
		pixsize (float, optional) : pixelsize in arcsec. 

		Attributes
		----------
		data (2d np array)
		pixsize (float)
		nx (int): image dimension
		ny (int): image dimension
		xc (int): image center
		yc (int): image center
		"""
		if 'filename' in kwargs:
			self.filename = kwargs.pop('filename', None)
			self.data = fits.getdata(self.filename)
			self.header = fits.getheader(self.filename)
		elif 'fn' in kwargs:
			self.filename = kwargs.pop('fn', None)
			self.data = fits.getdata(self.filename)
			self.header = fits.getheader(self.filename)
		elif 'data' in kwargs:
			self.data = kwargs.pop('data', None)
		else: 
			raise Exception('imgobj requires argument of either filename or data')

		self.pixsize = kwargs.pop('pixsize', None)
		self.ny, self.nx = self.data.shape
		self.yc, self.xc = self.ny//2, self.nx//2
