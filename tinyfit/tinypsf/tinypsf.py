"""
tinypsf uses tinytim to generate hst psf model in the flt frame. 
"""

import os
import numpy as np
from astropy.io import fits
from astropy.io import ascii
import json

from . import call_tinytim
from ..imgobj import imgobj


class tinypsf(object):
	"""Use tinytim to create HST PSF model in the flt frame. 

	Example: 
		>>> t = tinypsf(dir_out='./', fn='my_object', camera='wfc3_ir', filter='f160w', position=[563, 561], spectrum_form='stellar', spectrum_type='f8v', diameter=3, focus=-0.5, subsample=1)
		>>> status = t.make_psf()
		>>> t.psf.data
		>>> t.psf.writeto('psf.fits')

	Note:
		It has only been tested for ACS and WFC3. 
		The produced psf is named: 
			dir_out+fn+'.fits'
	Requirement:
		Tinytim 7.5 installed and linked. The environmental variable TINYTIM has to be set to the directory where tinytim is. 

	Attributes:
		dir_out
		fp_psf
		fn
		camera
		filter
		position
		spectrum_form
		spectrum_type
		focus
		subsample

	Methods:
		make_psf()
		get_psf()

	"""
	def __init__(self, camera='wfc3_ir', detector=0, filter='f160w', position=[500, 500], spectrum_form='stellar', spectrum_type='f8v', diameter=5, focus=0., subsample=1, dir_out='./', fn='psf_temporary', ): 
		"""initiate psf 
		Params:
			camera = 'wfc3_ir' (str)
				e.g., 'wfc3_ir'
			detector = 0 (int)
				e.g., 1 or 2. Only for when camera == 'acs_widefield'.
			filter = 'f160w' (str)
				e.g., 'f160w'
			position = [500, 500] (list)
				The position of the target in the _flt.fits file, e.g., [567, 789]	
			spectrum_form = 'stellar' (str)
				'stellar', 'blackbody', 'powerlaw_nu', 'powerlaw_lam', or 'user'
			spectrum_type = 'f8v' (str)
				input depends on spectrum_form
			diameter = 5 (float)
				diameter of the psf in arcsec
			focus = 0. (float)
				Focus, secondary mirror despace? [microns]
			subsample = 1 (int)
				Subsampling parameter. If it's larger than 1, then the final psf is oversampled by this factor. 
			dir_out = './' (str)
			fn = 'psf_temporary' (str)
				Rootname fn for the produced file with no extension. The output psf is dir_out+fn+'.fits'
		"""

		# tiny params
		self.camera = camera
		self.detector = detector
		self.filter = filter
		self.position = position
		self.spectrum_form = spectrum_form
		self.spectrum_type = spectrum_type
		self.diameter = diameter
		self.focus = focus
		self.subsample = subsample

		# filename
		self._set_filename(dir_out=dir_out, fn=fn, )

		# the environmental varaible to attribute 
		self.dir_tinytim = os.environ['TINYTIM']+'/'


	def write_params(self, fn='tiny_params.json'): 
		""" write the current setting to json file

		Args: 
			fn='tiny_params.json' (str): file to save to.  
		"""
		keys = ['fn', 'camera', 'detector', 'filter', 'position', 'spectrum_form', 'spectrum_type', 'diameter', 'focus', 'subsample']

		params = {key: getattr(self, key) for key in keys}

		with open(fn, 'w') as f:
			json.dump(params, f, indent='\t')


	def _set_filename(self, dir_out='./', fn='psf_temporary', ):
		self.dir_out = dir_out
		self.fn = os.path.splitext(fn)[0]
		self.rootname = self.dir_out+self.fn
		self.fp_param = self.dir_out+self.fn+'.param'
		self.fp_psf = self.dir_out+self.fn+'.fits'



	def make_psf(self):
		"""
		Call tinytim to produce psf. 

		Params
		------
		None

		Return
		------
		stauts (bool)

		Output
		------
		writes psf fits file to path:
			self.fp_psf

		along with other intermediate products:

			self.rootname+'.param'
			self.rootname+'.tt3'
			self.rootname+'00_psf.fits' (only for some cameras)
		"""
		if not os.path.isdir(self.dir_out):
			os.makedirs(self.dir_out)

		status_tiny = False

		ntry = 5
		i = 0
		while status_tiny == False and i < ntry:
			i += 1
			status1 = call_tinytim.tiny1(dir_code=self.dir_tinytim, fn=self.fp_param, camera=self.camera, detector=self.detector, position=self.position, filter=self.filter, spectrum_form=self.spectrum_form, spectrum_type=self.spectrum_type, diameter=self.diameter, focus=self.focus, rootname=self.rootname) 

			status2 = call_tinytim.tiny2(dir_code=self.dir_tinytim, fn=self.fp_param, rootname=self.rootname) 

			# for those camera that require tiny3 to do distortion and diffusion. 
			tiny3_cameras = ['acs_widefield', 'acs_highres', 'acs_coronoffspot', 'acs_solarblind', 'wfc3_uvis', 'wfc3_ir']

			if self.camera in tiny3_cameras:
				status3 = call_tinytim.tiny3(dir_code=self.dir_tinytim, fn=self.fp_param, rootname=self.rootname, subsample=self.subsample)
				if self.subsample > 1: 
					print("NOTE : Subsampled, so not convolving with charge diffusion kernel. Additional convolution required. ")
				status_tiny = np.all([status1, status2, status3])
			else: 
				status_tiny = np.all([status1, status2, ])

		if status_tiny: 
			os.rename(self.rootname+'00.fits', self.fp_psf)
			self._load_psf()
			return True
		else: 
			raise Exception("PSF not created properly")


	def get_psf(self):
		"""read and return psf np array

		Args:
			None

		Return:
			psf (np array)
		"""
		if not os.path.isfile(self.fp_psf):
			self.make_psf()

		return fits.getdata(self.fp_psf)


	def _load_psf(self):
		"""Load psf and its meta data from file to create object self.psf, which is an imgobj object that contains: 

		self.psf.data
		self.psf.pixsize
		self.psf.subsample
		self.psf.require_diffusion
		self.psf.diffusion_kernel

		Some of the oversampled psf required difussion after resampled to normal pixsize, in which cases self.psf.require_diffusion is True and the kernel for the convolution is in self.psf.diffusion_kernel. 

		Args:
			None

		Return:
			status (bool)
		"""

		hdus = fits.open(self.fp_psf)
		data = hdus[0].data
		header = hdus[0].header
		pixsize = header['PIXSCALE']

		self.psf = imgobj(data=data, pixsize=pixsize)
		self.psf.subsample=self.subsample
		self.psf.header = header

		if 'COMMENT' in header:
			sub_phrase = 'This PSF is subsampled, so the charge diffusion kernel'
			self.psf.require_diffusion = sub_phrase in header['COMMENT'][0]
		else: 
			self.psf.require_diffusion = False

		if self.psf.require_diffusion:
			tab = ascii.read(header['COMMENT'][3:6])
			self.psf.diffusion_kernel = np.array([tab[col] for col in tab.colnames])
			assert self.psf.diffusion_kernel.shape == (3, 3)

		else: 
			self.psf.diffusion_kernel = None

		status = type(self.psf) is imgobj
		return status
