# -*- coding: utf-8 -*-

"""
imgfitter takes in psf model and fit it to images. 
"""
import numpy as np
import scipy.optimize as so
import scipy.ndimage as sn
import copy
from collections import namedtuple
import json
from astropy.io import fits
import photutils as pu
from astropy import modeling as am
from astropy import convolution as ac



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
		self.chisq = None
		self.success = False
		self.isgal = False
		self.ishyper = False
		self.params = None
		self.galparams = None
		self.hyperparams = None


	def load(self, filename):
		""" load result from json file. 
		Args: 
			filename ('str')
		"""
		with open(filename, 'r') as file:
			result_dict = json.load(file)

		self.chisq = result_dict['chisq']
		self.success = result_dict['success']
		self.ishyper = result_dict['ishyper']
		try:
			self.isgal = result_dict['isgal']
		except: 
			self.isgal = False


		keys = list(result_dict['params'].keys())
		values = [result_dict['params'][key] for key in keys]
		self._set_params(params=values, names=keys)
		keys = list(result_dict['hyperparams'].keys())
		values = [result_dict['hyperparams'][key] for key in keys]
		self._set_hyperparams(params=values, names=keys)
		if self.isgal:
			keys = list(result_dict['galparams'].keys())
			values = [result_dict['galparams'][key] for key in keys]
			self._set_galparams(params=values, names=keys)


	def save(self, filename):
		""" save result as json file to filename. 
		Args:
			filename ('str')
		"""
		result_dict = {'params': dict(self.params._asdict()),
						'chisq': float(self.chisq), 
						'success': self.success,
						'isgal': self.isgal,
						'ishyper': self.ishyper,
						}
		if self.galparams is not None:
			result_dict.update({'galparams': dict(self.galparams._asdict()),})
		if self.hyperparams is not None: 
			result_dict.update({'hyperparams': dict(self.hyperparams._asdict()),})

		with open(filename, 'w') as file:
			json.dump(result_dict, file, indent=4)


	def _set_params(self, params, names):
		"""
		Attr: 
			params (:obj: 'np.ndarray')
			names (:obj: 'list')
		"""
		Params = namedtuple('Params', names)
		self.params = Params(*params)


	def _set_galparams(self, params, names):
		"""
		Attr: 
			params (:obj: 'np.ndarray')
			names (:obj: 'list')
		"""
		Params = namedtuple('Params', names)
		self.galparams = Params(*params)
		self.isgal = True


	def _set_hyperparams(self, params, names):
		"""
		Attr: 
			params (:obj: 'np.ndarray')
			names (:obj: 'list')
		"""
		Params = namedtuple('Params', names)
		self.hyperparams = Params(*params)
		self.ishyper = True

	def _set_params_from_OptimizeResult(self, optimizeresult, names):
		"""set attributes based on the scipy optimize OptimizeResult input. 

		Attr: 
			optimizeresult (:obj:'so.OptimizeResult')
			names (:obj: 'list' of 'str')
		"""
		params = _ensure_1d_array(optimizeresult.x)
		self._set_params(params=params[:len(names)], names=names)
		self.chisq = optimizeresult.fun
		self.success = optimizeresult.success


	def _set_galparams_from_OptimizeResult(self, optimizeresult, names, nskip=0):
		""" set attributes self.galparams based on the OptimizeResult input. The first 'nskip' params are ignored as they should belong to params, not galparams. 
		"""
		params = _ensure_1d_array(optimizeresult.x)
		self._set_galparams(params=params[nskip:], names=names)
		self.chisq = optimizeresult.fun
		self.success = optimizeresult.success


	def _set_hyperparams_from_OptimizeResult(self, optimizeresult, names):
		"""set attributes based on the scipy optimize OptimizeResult input. 

		Attr: 
			optimizeresult (:obj:'so.OptimizeResult')
			names (:obj: 'list' of 'str')
			mode = 'params' ('str'): either 'params' or 'hyperparams'. 
		"""
		params = _ensure_1d_array(optimizeresult.x)
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
		hypermodel_new._set_filename(dir_out=hypermodel_init.dir_out, fn=fn_new)

		hypermodel_new.make_psf()
		return hypermodel_new.psf

	else:
		raise Exception('The type of hypermodel is not implemented. Currently supporting only tinypsf. ')
		return None


def _params_in_range(params_dict, params_range):
	#  return True if params are all within range. 
	for key, rnge in params_range.items():
		if (rnge[0] is not None):
			if (params_dict[key] < rnge[0]):
				return False
		if (rnge[1] is not None):
			if (params_dict[key] > rnge[1]):
				return False
	return True


def _calc_chisq(imgdata, modeldata, model_maxlim=np.inf, neg_penal=1., largechisq=1.e10):
	"""
	calculate the chisq given two arrays -- imgdata and modeldata. model_maxlim is the maximum the model data can take, otherwise it will return inf chisq. neg_penal is a factor that penalizes negative residual. 
	"""
	if model_maxlim == 'data':
		maxlim = imgdata.max()
	else: 
		maxlim = model_maxlim

	# if model maximum goes beyond limit then return a large number
	if modeldata.max() > maxlim:
		return largechisq
	else: 
		if neg_penal == 1.:
			return np.nansum((imgdata - modeldata)**2)
		else:
			diff = imgdata - modeldata
			diff_pos = diff[diff>=0]
			diff_neg = diff[diff<0]
			return np.nansum((diff_pos)**2) + neg_penal**2 * np.sum((diff_neg)**2)


class imgfitter(object):
	""" `imgfitter` fits model to images. 

	Note:
		Image center is set to [ny//2, nx//2]. Counting starts from 0. 

	Examples:
		

		To fit the model to the image one can use fits files as input of the image and models. Here is how to fit the x, y shifting and a constant scaling to the model. Rebinning is taken care of as long as the image pixsize is a multiple of the model pixsize. Charge diffusion would be applied if it's required by the model. 

		>>> from tinyfit.imgfitter import imgfitter
		>>> f = imgfitter(filename='science_image.fits', pixsize=0.13, nx=64, ny=64)
		>>> f.set_model(filename='psf.fits', pixsize=0.026)
		>>> status = f.fit(x=512, y=511)

		>>> print(f.results.chisq)
		>>> print(f.results.params)
		>>> f.img_crop_bestfit.writeto('bestfit.fits')
		>>> f.img_crop_residual.writeto('residual.fits')

		Charge diffusion will be implemented if both of the followings are True. 

		>>> f.charge_diffusion
		>>> f.model_init.require_diffusion

		Fitting tinypsf model to image. 

		>>> from tinyfit.tinypsf import tinypsf
		>>> tpsf = tinypsf(camera='wfc3_ir', filter='f160w', position=[512, 511], 
						 spectrum_form='stellar', spectrum_type='k7v', diameter=6, 
						 focus=-0.5, subsample=5, fn='my_object', dir_out='./')

		>>> f = imgfitter(filename='science_image.fits', pixsize=0.13, nx=64, ny=64)
		>>> f.set_model(tinypsf=tpsf)
		>>> status = f.fit(x=512, y=511)

		>>> print(f.results.chisq)
		>>> print(f.results.params)
		>>> f.img_crop_bestfit.writeto('bestfit.fits')
		>>> f.img_crop_residual.writeto('residual.fits')

		To get bestfit of the dimension of the original image
		>>> f.img_bestfit.writeto('bestfit.fits')

		To get bestfit fits hdus with all the headers and extensinos taken from the original image. 
		>>> hdus_bestfit = f.get_hdus_bestfit()
		>>> hdus_bestfit.writeto('bestfit_hdus.fits')


		If in addition to fitting the x, y shift and scaling, one wants to fit tinypsf paramsters, such as the focus. `hyperfit()` takes significantly longer time than `fit()`. 

		>>> status = f.hyperfit(x=512, y=511, hyperparams=['focus'])
		>>> print(f.results.params)
		>>> print(f.results.hyperparams)

		To fit Gaussian smoothing in addition to x, y shift and scaling: 

		>>> status = f.fit(x=512, y=511, params=['dx', 'dy', 'scale', 'sigma'])
		>>> print(f.results.params)

		To fit Gaussian smoothing in addition to x, y shift and scaling and focus: 

		>>> status = f.fit(x=512, y=511, params=['dx', 'dy', 'scale', 'sigma'], hyperparams=['focus'])
		>>> print(f.results.params)
		>>> print(f.results.hyperparams)


	Attributes:
		nx = 64 (int): x dimension of the cropped image on which fitting is performed. 
		ny = 64 (int): y dimension of the cropped image on which fitting is performed. 
	    img (:obj:'imgobj'): imgobj object of the image to be fitted to. 
	    img_crop (:obj:'imgobj', optional): imgobj object of the cropped image.  
	    model_init  (:obj:'imgobj'): imgobj object of the model to fit to img_crop. 
	    hypermodel_init  (:obj:'tinypsf'): Either None or hyper model object. 
	    charge_diffusion (bool): if True, then charge diffusion is applied when needed. 
	    result (:obj:'_resultobj'): fitting results. 


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


	def set_galmodel(self, galmodel):
		""" set galaxy model

		Args:
			galmodel (:obj: 'astropy.modeling.models')
		"""
		self.galmodel = galmodel


	def fit(self, x=None, y=None, freeparams=['dx', 'dy', 'scale', 'background'], params_range={}, charge_diffusion=True, model_maxlim=np.inf, neg_penal=1., fitgal=False, galconvpsf=False, padbackground=True, uncrop_bestfit=True):
		""" Fit the model to the image at the specified location and image size. 

		Example:
			status = self.fit(x=500, y=500)

		Args: 
			x=None (int): x coordinate of the center of the cropped image
			y=None (int): y coordinate of the center of the cropped image
			freeparams=['dx', 'dy', 'scale', 'background'] (:obj: 'list' of 'str'): 
				List of free params to fit. Could be subset of: 
					['dx', 'dy', 'scale', 'background', 'sigma']. 
				'dx', 'dy': translation; 
				'scale': multiplied to pix values; 
				'background': constant level of background
				'sigma': width of Gaussian smoothing. By default no smoothing is fitted. 
			params_range={} (:obj: 'dictionary' of 'tuple')
				the range the free parameter can take, for example: {'dx': (-1., 1.), 'dy': (-2., 2.), }. Default: no range. If fitgal then galaxy model params can also be set via params_range. Alternatively, it can be specified as the bounds attribute of the astropy model. 
			charge_diffusion = True (int): 
				If true then apply charge diffusion to model if required. 
			model_maxlim = np.inf (float or str):
				Limits to the maximum value of the model image. Default: inf. Can be set to a number. If set to 'data', then it's determined from the max of data cropped image 
			neg_penal = 1. (float):
				Penalizing factor of the negative residuals in the chisq calculation. If set to larger than 1 then negative residuals are penalized relative to positive residuals. 
			fitgal = False (bool):
				If True, fit galaxy model simultaneously. The model is set by self.set_galmodel() as astropy.modeling.models. 
			galconvpsf=False (bool): 
				If true, the galaxy model is convolved with the PSF. 
			padbackground=True (bool):
				If true, the best fit uncropped image will be padded with bestfit background value. 
			uncrop_bestfit=True (bool):
				If True, attribute self.img_bestfit is set to be the uncropped bestfit image. 

		Return:
	        bool: The return value. True for success, False otherwise.

		Note:
			The image is first cropped at location [y, x] with sizes [ny, nx]. The model is regridded to match that of the cropped image. 

			By default, charge diffusion will be applied if required. 

			galconvpsf will significantly increase run time. It is implemented by convolving the galaxy model by a Gaussian kernel that best fits the PSF. 
		"""
		# sanity check
		if self.model_init is None:
			raise Exception("model_init does no exist")

		# set charge_diffusion
		self.charge_diffusion = charge_diffusion

		# cropping
		if self.img_crop is None:
			self._crop(xc=x, yc=y)

		# galmodel
		if fitgal:
			galmodel = self.galmodel
		else: 
			galmodel = None

		# fitting shift and scale
		res_xys = self._fitloop_xys(image=self.img_crop, psfmodel=self.model_init, psffreeparams=freeparams, params_range=params_range, model_maxlim=model_maxlim, neg_penal=neg_penal, galmodel=galmodel, galconvpsf=galconvpsf)

		if res_xys.success:

			self.result._set_params_from_OptimizeResult(res_xys, names=freeparams)
			if fitgal: 
				self.result._set_galparams_from_OptimizeResult(res_xys, names=galmodel.param_names, nskip=len(freeparams))
				self.result.isgal = True
			self.result._set_hyperparams(params=[], names=[])
			self.result.ishyper = False

			self.img_crop_bestfit = self.get_img_crop_bestfit()
			self.img_crop_residual = self.get_img_crop_residual(bestfit=self.img_crop_bestfit)
			if uncrop_bestfit:
				self.img_bestfit = self._uncrop(self.img_crop_bestfit, padbackground=padbackground)

			return True
		else: 
			return False


	def _fitloop_xys(self, image, psfmodel, psffreeparams=['dx', 'dy', 'scale', 'background'], params_range={}, model_maxlim=np.inf, neg_penal=1., galmodel=None, galconvpsf=False):
		""" loop for fitting psfmodel to image with free params x, y, scale 

		Args:
			image (:obj: imgobj)
			psfmodel (:obj: imgobj)
			psffreeparams=['dx', 'dy', 'scale', 'background'] (:obj: 'list' of 'str'): 
				List of free params to fit. Could be subset of: 
					['dx', 'dy', 'scale', 'background', 'sigma']. 
				'dx', 'dy': translation; 
				'scale': multiplied to pix values; 
				'background': constant level of background
				'sigma': width of Gaussian smoothing. By default no smoothing is fitted. 
			params_range={} (:obj: 'dictionary' of 'tuple')
				the range the free parameter can take, for example: {'dx': (-1., 1.), 'dy': (-2., 2.), }. Default: no range. If fitgal then galaxy model params can also be set via params_range. Alternatively, it can be specified as the bounds attribute of the astropy model. 
			model_maxlim = np.inf (float or str):
				Limits to the maximum value of the psfmodel image. Default: inf. Can be set to a number. If set to 'data', then it's determined from the max of data cropped image 
			neg_penal = 1. (float):
				Penalizing factor of the negative residuals in the chisq calculation. If set to larger than 1 then negative residuals are penalized relative to positive residuals. 
			galmodel=None (:obj: 'astropy.modeling.model'):
				galaxy model to be fitted simultaneously. All of it's parameters as specified by galmodel.param_names will be used as free parameters. 
			galconvpsf=False (bool): 
				If true, the galaxy model is convolved with the PSF. 

		return: 
			(:obj: scipy minimization result)
		"""

		def _chisq_xys(p, image, psfmodel, psffreeparams, params_range, model_maxlim=np.inf, neg_penal=1., galmodel=None, galconvpsf=False, largechisq=1.e10):
			""" calculate chisq. Return inf if psfmodel max is larter than model_maxlim. """
			# organize params and params_range
			psfparams_dict = dict(zip(psffreeparams, p[0:len(psffreeparams)]))
			if galmodel is None:
				params_dict = psfparams_dict
			else:
				params_dict = dict(zip(psffreeparams+list(galmodel.param_names), p))
				galmodel.parameters = np.array(p[len(psffreeparams):])
				for key in galmodel.bounds:
					if key not in params_range:
						params_range.update({key: galmodel.bounds[key]})
			if not _params_in_range(params_dict, params_range):
				return largechisq

			# transform psf
			psfmodel_new = self._get_shifted_scaled_smoothed_resampled_model(psfmodel, **psfparams_dict)
			assert psfmodel_new.pixsize == image.pixsize

			# evaluate galaxy model
			if galmodel is None: 
				modeldata = psfmodel_new.data
			else: 
				if galconvpsf: # convolve the 2d gaussian bestfit to psf
					if 'background' in params_dict:
						background = params_dict['background']
					else: 
						background = 0. 
					try:
						psf_normed = psfmodel_new.data-background
						psf_normed = psf_normed/psf_normed.max()
						gaussmodel = self._get_bestfit_gaussian_model(psf_normed, boxsize=16)
						galmodel_conv = ac.convolve_models(galmodel, gaussmodel, normalize_kernel=True, normalization_zero_tol=1e-8)
						galdata = self._evaluate_model(galmodel_conv)
					except: 
						print('Convolving galaxy with gaussian PSF failed, substituting PSF with default profile. ')
						gaussmodel = am.models.Gaussian2D(amplitude=1., x_mean=0., y_mean=0., x_stddev=1., y_stddev=1., theta=0.)
						galmodel_conv = ac.convolve_models(galmodel, gaussmodel, normalize_kernel=True, normalization_zero_tol=1e-8)
						galdata = self._evaluate_model(galmodel_conv)

				else: 
					galdata = self._evaluate_model(galmodel)
				modeldata = psfmodel_new.data+galdata

			chisq = _calc_chisq(imgdata=image.data, modeldata=modeldata, model_maxlim=model_maxlim, neg_penal=neg_penal)
			return chisq

		# set init
		init_dictionary = {'dx': 0.1, 'dy': 0.1, 
							# 'scale': image.data.max()/psfmodel.data.max(), 
							'scale': 100., 
							'sigma': 0.1, 'background': 0.}

		# reset init according to parmas_range
		for key, rnge in params_range.items():
			if key in psffreeparams:
				if (init_dictionary[key] < rnge[0]) | (init_dictionary[key] > rnge[1]):
					init_dictionary[key] = (rnge[0] + rnge[1])/2

		# the first len(freeparam) numbers are for psfmodel, the rest are for galmodel. 
		p_init = [init_dictionary[freeparam] for freeparam in psffreeparams]
		if galmodel is not None:
			p_init += list(galmodel.parameters)

		# run
		minresult = so.minimize(_chisq_xys, x0=p_init, args=(image, psfmodel, psffreeparams, params_range, model_maxlim, neg_penal, galmodel, galconvpsf), method='Powell')

		if 'sigma' in psffreeparams:
			isigma = np.where('sigma' == np.array(psffreeparams))[0][0]
			minresult.x[isigma] = np.absolute(minresult.x[isigma])

		return minresult


	def _get_bestfit_gaussian_image(self, data):
		""" return imgobj of the best fit gaussian to data """
		gaussmodel = self._get_bestfit_gaussian_model(data)
		y, x = self._yxmesh()
		return imgobj(data=gaussmodel(y, x), pixsize=self.img.pixsize)


	def _get_bestfit_gaussian_model(self, data, boxsize=None):
		""" return the best fit gaussian model to the cropped image data. 

		Note: 
			model in on meshgrid define by self_yxmesh()

		Args: 
			data (2d np array): of size of cropped image
			boxsize=None (int): 
				if set to integer then only the center box of diameter boxsize is used for the fit. 

		Return 
			(:obj: astropy model)
		"""
		if boxsize is not None:
			nyd, nxd = data.shape
			data_box = data[nyd//2-boxsize//2: nyd//2+boxsize//2, nxd//2-boxsize//2: nxd//2+boxsize//2]
		else: 
			data_box = copy.deepcopy(data)

		ny, nx = data_box.shape
		gaussmodel = am.models.Gaussian2D(amplitude=data_box.max(), x_mean=0., y_mean=0., x_stddev=2., y_stddev=2., theta=0.)
		gaussmodel.bounds['amplitude'] = (data_box.max()*1.e-3, None)
		gaussmodel.bounds['x_stddev'] = (0.1, None)
		gaussmodel.bounds['y_stddev'] = (0.1, None)
		y, x = self._yxmesh(ny=ny, nx=nx)
		fitter = am.fitting.LevMarLSQFitter()
		gaussmodel = fitter(gaussmodel, y, x, data_box)
		return gaussmodel


	def hyperfit(self, x, y, freeparams=['dx', 'dy', 'scale'], freehyperparams=[], params_range={}, charge_diffusion=True, model_maxlim=np.inf, neg_penal=1.):
		""" Fit the hypermodel to the image at the specified location and image size. 

		Example:
			status = self.hyperfit(x=500, y=500, freehyperparams=['focus', ])

		Args: 
			x (int): x coordinate of the center of the cropped image
			y (int): y coordinate of the center of the cropped image
			freeparams=['dx', 'dy', 'scale'] (:obj: 'list' of 'str'): 
				List of free params to fit. Could be subset of ['dx', 'dy', 'scale', 'sigma']. See fit() for details. 
			freehyperparams = [] (list: 'str'): 
				List of strings indicating hyperparameters to fit, e.g., ['focus', ]. 
			params_range={} (:obj: 'dictionary' of 'tuple')
				the range the free parameter can take, for example: {'dx': (-1., 1.), 'dy': (-2., 2.), }. Default: no range. 
			charge_diffusion = True (int): 
				If true then apply charge diffusion to model if required. 
			model_maxlim = np.inf (float or str):
				Limits to the maximum value of the model image. Default: inf. Can be set to a number. If set to 'data', then it's determined from the max of data cropped image 
			neg_penal = 1. (float):
				Penalizing factor of the negative residuals in the chisq calculation. If set to larger than 1 then negative residuals are penalized relative to positive residuals. 

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
		self._crop(xc=x, yc=y)

		res_hyper, res_xys = self._fitloop_hyperparam(image=self.img_crop, hypermodel_init=self.hypermodel_init, freeparams=freeparams, freehyperparams=freehyperparams, params_range=params_range, model_maxlim=model_maxlim, neg_penal=neg_penal)

		if res_hyper.success and res_xys.success:
			print('hyperfit successful')

			# import pdb; pdb.set_trace()
			self.result._set_params_from_OptimizeResult(res_xys, names=freeparams)
			self.result._set_hyperparams_from_OptimizeResult(res_hyper, names=freehyperparams)

			self.img_crop_bestfit = self.get_img_crop_bestfit()
			self.img_crop_residual = self.get_img_crop_residual(bestfit=self.img_crop_bestfit)
			self.img_bestfit = self._uncrop(self.img_crop_bestfit)

			return True
		else: 
			print('hyperfit failed')
			return False


	def _fitloop_hyperparam(self, image, hypermodel_init, freeparams, freehyperparams, params_range={}, model_maxlim=np.inf, neg_penal=1.):
		""" fitting hypermodel to image with free hyperparams

		Args:
			image (:obj: imgobj)
			model (:obj: imgobj)
			hypermodel_init (:obj: tinypsf)
			freeparams (:obj: 'list' of 'str')
			freehyperparams (:obj: 'list' of 'str')
			params_range={} (:obj: 'dictionary' of 'tuple')
			model_maxlim = np.inf (float or str)
			neg_penal = 1. (float)

		return: 
			(:obj: scipy minimization result)
		"""

		def _chisq_hyper(p, freeparams, freehyperparams, image, hypermodel_init, params_range, model_maxlim=np.inf, neg_penal=1., largechisq=1.e10):
			""" p = (dx, dy, scale) """
			hyperparams_dict = dict(zip(freehyperparams, p))
			model_new = _make_model_with_new_hyperparam(hyperparams_dict, hypermodel_init)

			res_xys = self._fitloop_xys(image, model_new, psffreeparams=freeparams, params_range=params_range, model_maxlim=model_maxlim, neg_penal=neg_penal)

			params_dict = dict(zip(freeparams, res_xys.x))
			if not _params_in_range(params_dict, params_range): 
				return largechisq

			model_xyss = self._get_shifted_scaled_smoothed_resampled_model(model_new, **params_dict)
			assert model_xyss.pixsize == image.pixsize

			return _calc_chisq(imgdata=image.data, modeldata=model_xyss.data, model_maxlim=model_maxlim, neg_penal=neg_penal)

		# hyperfit
		params_init = [getattr(hypermodel_init, h) for h in freehyperparams]
		res_hyper = so.minimize(_chisq_hyper, x0=params_init, args=(freeparams, freehyperparams, image, hypermodel_init, params_range, model_maxlim, neg_penal), method='Powell')

		# pick up xys of the bestfit hyperfit
		params_dict = dict(zip(freehyperparams, _ensure_1d_array(res_hyper.x)))
		model_new = _make_model_with_new_hyperparam(params_dict, hypermodel_init)
		res_xys = self._fitloop_xys(image, model_new, psffreeparams=freeparams, params_range=params_range, model_maxlim=model_maxlim, neg_penal=neg_penal)

		return res_hyper, res_xys


	def get_hdus_bestfit(self, padbackground=True):
		"""
		Return the hdus that is taken from the input file but with science image replaced by the bestfit. 

		Args: 
			padbackground=True (bool): 
				If True, pad the uncovered pixels with bestfit background. 
		"""
		img_bestfit = self._uncrop(self.get_img_crop_bestfit(), padbackground=padbackground)
		hdus = fits.open(self.img.filename)
		hdus[1].data = img_bestfit.data
		hdus[0].header['HISTORY'] = 'TINYFIT: SCI IMAGE REPLACED BY BESTFIT IMAGE'
		hdus[1].header['HISTORY'] = 'TINYFIT: IMAGE REPLACED BY BESTFIT IMAGE'
		return hdus


	def get_hdus_bestfitpsf(self, padbackground=True):
		"""
		Return the hdus that is taken from the input file but with science image replaced by the bestfit. 

		Args: 
			padbackground=True (bool): 
				If True, pad the uncovered pixels with bestfit background. 
		"""
		img_bestfit = self._uncrop(self.get_img_crop_bestfitpsf(), padbackground=padbackground)
		hdus = fits.open(self.img.filename)
		hdus[1].data = img_bestfit.data
		hdus[0].header['HISTORY'] = 'TINYFIT: SCI IMAGE REPLACED BY BESTFIT PSF'
		hdus[1].header['HISTORY'] = 'TINYFIT: IMAGE REPLACED BY BESTFIT PSF'
		return hdus


	def get_img_crop_bestfitgal(self):
		""" return the imgobj of the current bestfit galaxy model """
		self.galmodel.parameters = np.array(list(self.result.galparams))
		galdata = self._evaluate_model(self.galmodel)
		return imgobj(data=galdata, pixsize=self.img_crop.pixsize)


	def get_img_crop_bestfitpsf(self):
		""" return the imgobj of the current bestfit psf model """
		if self.result.ishyper:
			model = _make_model_with_new_hyperparam(params_dict=self.result.hyperparams._asdict(), hypermodel_init=self.hypermodel_init)
		else: 
			model = self.model_init

		model_xyss = self._get_shifted_scaled_smoothed_resampled_model(model, **self.result.params._asdict())
		return model_xyss


	def get_img_crop_bestfit(self):
		""" Return the imgobj of the current model corresponding to the params in self.result. 
		"""
		model_psf = self.get_img_crop_bestfitpsf()

		if self.result.isgal:
			galdata = self.get_img_crop_bestfitgal().data
			model_psf.data += galdata

		return model_psf


	def get_img_crop_residual(self, bestfit=None):
		""" Return the imgobj of the cropped image minus the current bestfit model 
		Args: 
			bestfit (:obj: 'imgobj', optional)
		"""
		if bestfit is None:
			bestfit = self.get_img_crop_bestfit()
		assert bestfit.pixsize == self.img_crop.pixsize
		residual_data = self.img_crop.data - bestfit.data
		return imgobj(data=residual_data, pixsize=self.img_crop.pixsize)


	def find_cropcentroid(self, imgcoord=False, update=False):
		"""
		return centroid coordinate of cropped image in cropped image coordinate (xc_crop, yc_crop). 

		Args: 
			imgcoord (bool): if True, return in image coordinate (x, y) instead. 
			update (bool): if True, creating new crop based on the new coordinate. 

		Return 
			(float, float): image coordinate (x, y). 
		"""
		xc_crop_new, yc_crop_new = pu.centroid_1dg(self.img_crop.data)

		if not imgcoord:
			result =  xc_crop_new, yc_crop_new
		else: 
			result = self._cropxy_to_xy(xc_crop_new, yc_crop_new)

		if update:
			xc, yc = np.round(self._cropxy_to_xy(xc_crop_new, yc_crop_new)).astype(int)
			self._crop(xc, yc)

		return result


	def get_petrosian_radius(self):
		"""
		return petrosian radius of the data. 
		"""
		return imgtools.petrosian_radiu(self.img_crop.data, r_step=5., petrosian_ratio=0.2)


	def set_img_crop(self, **kwargs):
		""" set the attribute self.img_crop with data

		Args:
			**kwargs for imgobj. 
		"""
		self.img_crop = imgobj(pixsize=self.img.pixsize, **kwargs)
		assert self.img_crop.data.shape[0] == self.ny
		assert self.img_crop.data.shape[1] == self.nx


	def _crop(self, xc, yc):
		"""cropped the image and save to attribute self.img_crop. 

		Note: 
			The centroid of the cropped image (yc, xc) cannot be outside the original image. 
			If part of the cropped image falls outside of the original image, the outside edge will be padded with 0. 

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

		# check that the centroid of the cropped image is within self.img
		if not _pointxy_is_in_img(xc, yc, self.img): 
			raise Exception("cropping location outside image ")

		x0, x1, y0, y1 = imgtools.get_cutout_xy_range(xc, yc, self.nx, self.ny)

		if _pointxy_is_in_img(x0, y0, self.img) & _pointxy_is_in_img(x1, y1, self.img):
			# padding not required
			data_crop = self.img.data[y0:y1, x0:x1]
		else: 
			# padding required
			pad_width = ((self.ny, self.ny), (self.nx, self.nx))
			img_pad = np.pad(self.img.data, pad_width=pad_width, mode='constant', constant_values=0.)
			data_crop = img_pad[y0+self.ny: y1+self.ny, x0+self.nx: x1+self.nx]

		self.img_crop = imgobj(data=data_crop, pixsize=self.img.pixsize)


	def _uncrop(self, img_crop, xc=None, yc=None, padbackground=True):
		"""
		Return an uncropped image from a cropped image. The uncovered pixels are set to zero. 

		Args: 
			img_crop (:obj:'imgobj'): imgobj object to be uncropped
			xc=None (int, optional): x center of the cropped image. Default self._cropxc.
			yc=None (int, optional): y center of the cropped image. Default self._cropxc.
			padbackground=True (bool): 
				If True, pad the uncovered pixels with bestfit background. 

		Return:			
			(:obj:'imgobj'): uncropped image of the same size as self.img
		"""
		if xc is None:
			xc = self._cropxc
		if yc is None:
			yc = self._cropyc

		if padbackground:
			if hasattr(self.result.params, 'background'): 
				offset = self.result.params.background
			else: 
				offset = 0. 
		else: 
			offset = 0. 

		# check that the centroid of the cropped image is within self.img
		if not _pointxy_is_in_img(xc, yc, self.img): 
			raise Exception("cropping location outside image ")

		x0, x1, y0, y1 = imgtools.get_cutout_xy_range(xc, yc, self.nx, self.ny)

		if _pointxy_is_in_img(x0, y0, self.img) & _pointxy_is_in_img(x1, y1, self.img):
			# padding not required
			data_uncrop = np.zeros(self.img.data.shape)+offset
			data_uncrop[y0:y1, x0:x1] = img_crop.data
		else: 
			# padding required
			data_uncrop = np.zeros(np.array(self.img.data.shape)+np.array([self.ny*2, self.nx*2]))+offset
			data_uncrop[y0+self.ny: y1+self.ny, x0+self.nx: x1+self.nx] = img_crop.data
			data_uncrop = data_uncrop[self.ny:-self.ny, self.nx: -self.nx]

		img_uncrop = imgobj(data=data_uncrop, pixsize=self.img.pixsize)
		return img_uncrop




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


	def _xy_to_cropxy(self, x, y):
		"""Translate x, y coordinate in the cropped image to that of the original image. 

		Args:
			x (float): x coordinate in the original image
			y (float): y coordinate in the original image

		Return: 
			x_crop (float): x coordinate in the cropped image
			y_crop (float): y coordinate in the cropped image			
		"""
		x_crop = x + self.img_crop.xc - self._cropxc 
		y_crop = y + self.img_crop.yc - self._cropyc

		return x_crop, y_crop


	def _evaluate_model(self, model):
		""" evaluate self.galmodel on the grid of cropped image and return image data """
		if self.galmodel is None: 
			raise Exception("galaxy model not set")

		y, x = np.meshgrid(np.arange(-self.ny//2, self.ny//2), np.arange(-self.nx//2, self.nx//2))
		return model(y, x)


	def _yxmesh(self, ny=None, nx=None):
		""" return y, x for meshgrid 

		Args: 
			ny=None (int): size of mesh, default: self.ny
			nx=None (int): size of mesh, default: self.nx
		"""
		if ny is None:
			ny = self.ny
		if nx is None:
			nx = self.nx
		y, x = np.meshgrid(np.arange(-ny//2, ny//2), np.arange(-nx//2, nx//2))
		return y, x

	def _get_shifted_scaled_smoothed_resampled_model(self, model, dx=0., dy=0., scale=1., background=0., sigma=0.):
		""" Shift, scale, smooth, and resample model to img_crop grids according to params. 
		"""
		if (dx!=0.) | (dy!=0.) | (scale!=1.) | (background!=0.):
			model_new = self._get_shifted_scaled_resampled_model(model, dx, dy, scale, background)
		else: 
			model_new = copy.copy(model)

		sigma = np.absolute(sigma)
		if sigma > 0.:
			model_new.data = sn.gaussian_filter(model_new.data, sigma=sigma)

		return model_new


	def _get_shifted_scaled_resampled_model(self, model, dx, dy, scale, background):
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
			background (float):
				a constant to be added to the background. 

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

		data_final += background
		# return
		model_final = imgobj(data=data_final, pixsize=self.img.pixsize)

		return model_final


def _pointxy_is_in_img(x, y, img):
	""" 
	return True if point (y, x) is in image with shape (ny, nx) then return, False otherwise.

	Args: 
		x (int)
		y (int)
		img (:obj: imgobj)
	"""
	nx, ny = img.nx, img.ny
	return (x > 0) & (x < nx) & (y > 0) & (y < ny)
