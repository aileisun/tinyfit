# -*- coding: utf-8 -*-

"""
imgfitter takes in psf model and fit it to images. 
"""
import numpy as np
import scipy.optimize as so
import scipy.ndimage as sn
import copy
from collections import namedtuple

from ..imgobj import imgobj
from ..tinypsf import tinypsf
from . import imgtools


def _ensure_1d_array(a):
	"""
	whether a is an 0d array or 1d array, the return will be 1d array
	"""
	if a.shape == ():
		return np.array([a])
	elif type(a) is np.ndarray and a.shape[0]>=1: 
		return a
	else: 
		raise Exception('input format is not np.ndarray')


class _resultobj():
	""" class that contains the results of the fit

	Attributes:
		params (:obj: 'collections.namedtuple')
		hyperparams (:obj: 'collections.namedtuple')
		chisq (float)
		success (bool)
		ishyper (bool)
	"""
	def __init__(self):
		self.params = None
		self.hyperparams = None
		self.chisq = None
		self.success = False
		self.ishyper = False

	def _set_params(self, params, names):
		"""
		Attr: 
			params (:obj: 'np.ndarray')
			names (:obj: 'list')
		"""
		Params = namedtuple('Params', names)
		self.params = Params(*params)


	def _set_hyperparams(self, params, names):
		"""
		Attr: 
			params (:obj: 'np.ndarray')
			names (:obj: 'list')
		"""
		Params = namedtuple('Params', names)
		self.hyperparams = Params(*params)
		self.ishyper = True

	def _set_from_OptimizeResult(self, optimizeresult, names, mode='params'):
		"""set attributes based on the scipy optimize OptimizeResult input. 

		Attr: 
			optimizeresult (:obj:'so.OptimizeResult')
			names (:obj: 'list' of 'str')
			mode = 'params' ('str'): either 'params' or 'hyperparams'. 
		"""


		# if optimizeresult.x.shape == ():
		# 	params = np.array([optimizeresult.x])
		# elif (type(optimizeresult.x) is np.ndarray) and optimizeresult.x.shape[0]>=1: 
		# 	params = optimizeresult.x
		# else: 
		# 	raise Exception('optimizeresult.x format not recognized. ')

		params = _ensure_1d_array(optimizeresult.x)

		if mode == 'params':
			self._set_params(params=params, names=names)
		else: 
			self._set_hyperparams(params=params, names=names)
			self.ishyper = True

		self.chisq = optimizeresult.fun
		self.success = optimizeresult.success


def _make_model_with_new_hyperparam(params_dict, hypermodel_init):
	"""
	return new model that has hyperparams changed. 

	Args:
		params_dict (dict): dictionary of input hyperparams
		hypermodel_init (:obj:): initial hypermodel 
	"""
	if type(hypermodel_init) is tinypsf:
		hypermodel_new = copy.copy(hypermodel_init)

		for key in params_dict:
			setattr(hypermodel_new, key, params_dict[key])

		fn_new = hypermodel_init.fn+'_hyperfit'
		hypermodel_new.set_filename(dir_out=hypermodel_init.dir_out, fn=fn_new)

		hypermodel_new.make_psf()
		return hypermodel_new.psf

	else:
		raise Exception('The type of hypermodel is not implemented. Currently supporting only tinypsf. ')
		return None


class imgfitter(object):
	""" `imgfitter` fits model to images. 

	Note:
		Image center is set to [ny//2, nx//2]. Counting starts from 0. 

	Attributes:
		nx = 64 (int): x dimension of the cropped image on which fitting is performed. 
		ny = 64 (int): y dimension of the cropped image on which fitting is performed. 
	    img (:obj:'imgobj'): imgobj object of the image to be fitted to. 
	    img_crop (:obj:'imgobj', optional): imgobj object of the cropped image.  
	    model_init  (:obj:'imgobj'): imgobj object of the model to fit to img_crop. 
	    hypermodel_init  (:obj:'tinypsf'): Either None or hyper model object. 
	    charge_diffusion (bool): if True, then charge diffusion is applied when needed. 
	    result (:obj:'_resultobj'): fitting results. 

	Examples:
		One can use fits files as input of the image and models. 

		>>> f = imgfitter(filename='science_image.fits', pixsize=0.13, nx=64, ny=64)
		>>> f.set_model(filename='psf.fits', pixsize=0.026)
		>>> status = f.fit(x=512, y=511)
		>>> f.results.chisq

		>>> img_crop_residual = f.get_img_crop_residual()
		>>> img_crop_bestfit.writeto('residual.fits')

	"""
	def __init__(self, nx=64, ny=64, charge_diffusion=True, **kwargs):
		""" Initialize imgfitter -- setting the image to be fitted to. 
		Args:
			filename (str): File path to fits file of the image to be fitted to. 
				-- or alternatively --
				fn (str): File path to fits file of the image to be fitted to. 
				data (:obj:'np.array'): 2d image array of floats. 
			pixsize (float, optional): pixelsize in arcsec. 
			nx = 64 (int): x dimension of the cropped image on which fitting is performed. 
			ny = 64 (int): y dimension of the cropped image on which fitting is performed. 
			charge_diffusion = True (int): 
				If true then apply charge diffusion to model if required. 

		"""
		self.nx = nx
		self.ny = ny
		self.img = imgobj(**kwargs)
		self.img_crop = None
		self.model_init = None
		self.hypermodel_init = None
		self.charge_diffusion = charge_diffusion
		self.result = _resultobj()


	def set_model(self, **kwargs):
		""" Setting the model by filename or data. 

		The model can be set by either a fits file, a data np array, or a tinypsf instance. To enable advanced fitting capability, please use tinypsf instance. 

		Args:
			filename (str): File path to fits file of the image to be fitted to. 
				-- or alternatively --
				fn (str): File path to fits file of the image to be fitted to. 
				data (:obj:'np.array'): 2d image array of floats. 
			pixsize (float, optional): pixelsize in arcsec. 

			-- or alternatively --
			tinypsf (:obj:tinypsf): instance of tinypsf

		Note:
			Sets attribute `model_init` as the imgobj of input model. 
			If tinypsf is used for initializing, besides setting model_init, 'hypermodel_init' is set to the input tinypsf object. 
		"""
		if ('filename' in kwargs) or ('fn' in kwargs) or ('data' in kwargs):
			self.model_init = imgobj(**kwargs)
		elif 'tinypsf' in kwargs:
			self.hypermodel_init = kwargs.pop('tinypsf', None)
			if not hasattr(self.hypermodel_init, 'psf'):
				self.hypermodel_init.make_psf()
			self.model_init = self.hypermodel_init.psf


	def fit(self, x, y, charge_diffusion=True):
		""" Fit the model to the image at the specified location and image size. 

		Example:
			status = self.fit(x=500, y=500)

		Args: 
			x (int): x coordinate of the center of the cropped image
			y (int): y coordinate of the center of the cropped image
			charge_diffusion = True (int): 
				If true then apply charge diffusion to model if required. 

		Return:
	        bool: The return value. True for success, False otherwise.

		Note:
			The image is first cropped at location [y, x] with sizes [ny, nx]. The model is regridded to match that of the cropped image. 

			By default, charge diffusion will be applied if required. 
		"""
		# sanity check
		if self.model_init is None:
			raise Exception("model_init does no exist")

		# set charge_diffusion
		self.charge_diffusion = charge_diffusion

		# cropping
		self._cropimg(xc=x, yc=y)

		# fitting shift and scale
		res_xys = self._fitloop_xys(image = self.img_crop, model=self.model_init)

		if res_xys.success:
			# set result attribute
			self.result._set_from_OptimizeResult(res_xys, names=['dx', 'dy', 'scale'], mode='params')
			self.result.ishyper = False

			self.img_crop_bestfit = self.get_result_model()
			assert self.img_crop_bestfit.pixsize == self.img_crop.pixsize
			data_residual = self.img_crop.data - self.img_crop_bestfit.data
			self.img_crop_residual = imgobj(data=data_residual, pixsize=self.img_crop.pixsize)

			return True
		else: 
			return False


	def hyperfit(self, x, y, hyperparams=[], charge_diffusion=True):
		""" Fit the hypermodel to the image at the specified location and image size. 

		Example:
			status = self.hyperfit(x=500, y=500, hyperparams=['focus', ])

		Args: 
			x (int): x coordinate of the center of the cropped image
			y (int): y coordinate of the center of the cropped image
			hyperparams = [] (list: 'str'): 
				List of strings indicating hyperparameters to fit, e.g., ['focus', ]. 
			charge_diffusion = True (int): 
				If true then apply charge diffusion to model if required. 

		Return:
	        bool: The return value. True for success, False otherwise.

		Note:
			see `fit()`. 
		"""
		# sanity check
		if self.model_init is None:
			raise Exception("model_init does no exist, please use set_model() to initialize.")
		if self.hypermodel_init is None:
			raise Exception("hypermodel_init does no exist, please use set_model() to initialize.")

		# set charge_diffusion
		self.charge_diffusion = charge_diffusion

		# cropping
		self._cropimg(xc=x, yc=y)

		res_hyper, res_xys = self._fitloop_hyperparam(image=self.img_crop, hypermodel_init=self.hypermodel_init, hyperparams=hyperparams)

		if res_hyper.success and res_xys.success:
			print('hyperfit successful')

			self.result._set_from_OptimizeResult(res_xys, names=['dx', 'dy', 'scale'], mode='param')
			self.result._set_from_OptimizeResult(res_hyper, names=hyperparams, mode='hyperparam')

			self.img_crop_bestfit = self.get_result_model()
			assert self.img_crop_bestfit.pixsize == self.img_crop.pixsize
			data_residual = self.img_crop.data - self.img_crop_bestfit.data
			self.img_crop_residual = imgobj(data=data_residual, pixsize=self.img_crop.pixsize)

		else: 
			print('hyperfit failed')
			return False


	def get_result_model(self):
		""" Return the imgobj of the current model corresponding to the params in self.result. 
		"""
		if self.result.ishyper:
			model = _make_model_with_new_hyperparam(params_dict=self.result.hyperparams._asdict(), hypermodel_init=self.hypermodel_init)
		else: 
			model = self.model_init

		model_xys = self._get_shifted_resampled_model(model, **self.result.params._asdict())
		return model_xys


	def _fitloop_xys(self, image, model):
		""" loop for fitting model to image with free params x, y, scale 

		Args:
			image (:obj: imgobj)
			model (:obj: imgobj)

		return: 
			(:obj: scipy minimization result)
		"""

		def _chisq_xys(p, image, model):
			""" p = (dx, dy, scale) """
			dx, dy, scale = p
			model_xys = self._get_shifted_resampled_model(model, dx, dy, scale)
			assert model_xys.pixsize == image.pixsize
			c = np.sum((image.data - model_xys.data)**2)
			return c

		params_init = [0., 0., image.data.max()/model.data.max()]
		minresult = so.minimize(_chisq_xys, x0=params_init, args=(image, model, ), method='Powell')

		return minresult


	def _fitloop_hyperparam(self, image, hypermodel_init, hyperparams):
		""" fitting hypermodel to image with free hyperparams

		Args:
			image (:obj: imgobj)
			model (:obj: imgobj)

		return: 
			(:obj: scipy minimization result)
		"""

		def _chisq_hyper(p, hyperparams, image, hypermodel_init):
			""" p = (dx, dy, scale) """
			params_dict = dict(zip(hyperparams, p))
			model_new = _make_model_with_new_hyperparam(params_dict, hypermodel_init)

			res_xys = self._fitloop_xys(image, model_new)

			dx, dy, scale = res_xys.x
			model_xys = self._get_shifted_resampled_model(model_new, dx, dy, scale)
			assert model_xys.pixsize == image.pixsize
			c = np.sum((image.data - model_xys.data)**2)
			return c

		# hyperfit
		params_init = [getattr(hypermodel_init, hyperparam) for hyperparam in hyperparams]
		res_hyper = so.minimize(_chisq_hyper, x0=params_init, args=(hyperparams, image, hypermodel_init), method='Powell')

		# pick up xys of the bestfit hyperfit
		params_dict = dict(zip(hyperparams, _ensure_1d_array(res_hyper.x)))
		model_new = _make_model_with_new_hyperparam(params_dict, hypermodel_init)
		res_xys = self._fitloop_xys(image, model_new)

		return res_hyper, res_xys


	def _cropimg(self, xc, yc):
		"""cropped the image and save to attribute self.img_crop. 

		Args:
			xc (int): x center of the cropped image
			yc (int): y center of the cropped image

		Note:
			Sets the following attributes:
		    img_crop (:obj:'imgobj'): imgobj object of the cropped image. 
			_cropxc (int): cetner of the cropped image in the original image. 
			_cropyc (int): 
		"""
		self._cropxc = xc
		self._cropyc = yc

		x0, x1, y0, y1 = imgtools.get_cutout_xy_range(xc, yc, self.nx, self.ny)
		data_crop = self.img.data[y0:y1, x0:x1]
		self.img_crop = imgobj(data=data_crop, pixsize=self.img.pixsize)


	def _cropxy_to_xy(self, x, y):
		"""Translate x, y coordinate in the cropped image to that of the original image. 

		Args:
			x (float): x coordinate in the cropped image
			y (float): y coordinate in the cropped image

		Return: 
			x_img (float): x coordinate in the original image
			y_img (float): y coordinate in the original image			
		"""
		x_img = x - self.img_crop.xc + self._cropxc
		y_img = y - self.img_crop.yc + self._cropyc

		return x_img, y_img


	def _get_shifted_resampled_model(self, model, dx, dy, scale, ):
		"""
		Shift and resample the input model to produce a output that has the same pixsize as the self.img and have dimensions [ny, nx]. 

		Args:
			model (:obj: imgobj): the input model. 
			dx (float):
				the shift of model in x direction in units of original image pixel. 
			dy (float):
				the shift of model in x direction in units of original image pixel. 
			scale (float):
				the scaling parameter to be multiplied to the model. 

		Return:
			:obj:'imgobj': imgobj object of the resampled model. 

		Note: 
			If there is no shift (dx, dy are zero), then the model will be centered on the new center [ny//2, nx//2]. Uncovered edges will be padded with zeros. The amount of shift is in units of the original image pixsize. 

			Currently resample only applies to subsampled models that have pixsize a integer fraction , e.g., 1/2, 1/5, etc., of the original image pixsize.

			Shift is applied to subsampled model with linear interpolation of pixel values, which perserves the flux. After interpolation, the subsampled model is regridded to the desired pixsize by integrating n*n pixels into one pixel, where n is the subsampling factor. In practice, it is done by convoluting the model with a 2d top-hat function of size n*n and resampling every n pixels. 

			If self.charge_diffusion is true then apply charge diffusion to model if required by self.model_require_diffusion. 
		"""

		# sanity check
		if model is None:
			raise Exception("model does no exist")
		if not (self.img.pixsize/model.pixsize).is_integer():
			raise Exception("Image pixel size is not a multiple of model pixel size. ")

		# shift resample
		subsample = int(self.img.pixsize/model.pixsize)

		data_shifted = imgtools.shift(model.data, dx, dy, subsample=subsample)
		data_resamp = scale*imgtools.resample(data_shifted, self.nx, self.ny, subsample=subsample)

		# charge diffusion
		if self.charge_diffusion and hasattr(model, 'require_diffusion'):
			if model.require_diffusion == True:
				data_final = sn.convolve(data_resamp, model.diffusion_kernel, mode='constant', cval=0.)
			else: 
				data_final = data_resamp
		else: 
			data_final = data_resamp

		# return
		model_final = imgobj(data=data_final, pixsize=self.img.pixsize)

		return model_final


