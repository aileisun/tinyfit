# -*- coding: utf-8 -*-

"""
imgfitter takes in psf model and fit it to images. 
"""

from .. import imgobj

from . import imgtools

class imgfitter(object):
	"""
	imgfitter fits model to images. It handles croping of stamps. 

	Note
	----
	Image center is set to [ny//2, nx//2]. Counting starts from 0. 

	Attributes
	----------
    img (:obj:'imgobj'): imgobj object of the image to be fitted to. 

	"""

	def __init__(self, **kwargs):
		"""
		Parameters
		----------
			
		filename (str): File path to fits file of the image to be fitted to. 
			-- or alternatively --
			fn (str): File path to fits file of the image to be fitted to. 
			data (:obj:'np.array'): 2d image array of floats. 
		pixsize (float, optional): pixelsize in arcsec. 

		"""
		self.img = imgobj(**kwargs)
		

	def _crop(self, xc, yc, nx, ny):
		"""
		make cropped image self.img_crop. 

		Parameters
		----------
		xc (int): center of the cropped image
		yc (int)
		nx (int): dimensino of the cropped image
		ny (int)

		Attributes to set
		-----------------
	    img_crop (:obj:'imgobj'): imgobj object of the cropped image. 
		_cropxc (int): cetner of the cropped image in the original image. 
		_cropyc (int): 
		"""
		self._cropxc = xc
		self._cropyc = yc


		self.img_crop = imgobj(data=data, pixelsize=self.img.pixelsize)


		imgtools.get_cutout_xy_range(xc, yc, nx, ny)