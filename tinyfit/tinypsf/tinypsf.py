#!/usr/bin/env python

"""
class tinypsf that can generate hst psf array in the flt image frame. 
"""

__author__ = "Ai-Lei Sun"
__copyright__ = "Copyright 2018, TinyFitter"

__license__ = "MIT"
__version__ = "0.0.0"
__maintainer__ = "Ai-Lei Sun"
__email__ = "aileisundr@gmail.com"
__status__ = "Prototype"


import os
from astropy.io import fits
import numpy as np

from . import call_tinytim


class tinypsf(object):
	def __init__(self, camera='wfc3_ir', detector=0, filter='f160w', position=[500, 500], spectrum_form='stellar', spectrum_type='f8v', diameter=5, focus=0., dir_out='./', fn='psf_temporary', ): 
		"""
		tinypsf (class)

		Calls tinytim to produce hst psf in flt frame given the input parameters. 

		The produced psf is named: 
			dir_out+fn+'.fits'

		Requirement
		-----------
		Tinytim 7.5 installed and linked. The environmental variable TINYTIM has to be set to the directory where tinytim is. 

		Params
		------
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
		dir_out = './' (str)
		fn = 'psf_temporary' (str)
			Rootname fn for the produced file with no extension. The output psf is dir_out+fn+'.fits'

		Attributes
		----------
		dir_out
		fp_psf
		fn
		camera
		filter
		position
		spectrum_form
		spectrum_type
		focus

		Methods
		-------
		make_psf()
		read_psf()
		"""

		self.dir_out = dir_out
		self.fn = os.path.splitext(fn)[0]
		self.rootname = self.dir_out+self.fn
		self.fp_param = self.dir_out+self.fn+'.param'

		self.camera = camera
		self.detector = detector
		self.filter = filter
		self.position = position
		self.spectrum_form = spectrum_form
		self.spectrum_type = spectrum_type
		self.diameter = diameter
		self.focus = focus

		# the output file fn of tinytim
		self.fp_psf = self.dir_out+self.fn+'.fits'

		# the environmental varaible to attribute 
		self.dir_tinytim = os.environ['TINYTIM']+'/'


	def make_psf(self, run_tiny3=False):
		"""
		Call tinytim to produce psf. 

		Params
		------
		run_tiny3=False (bool)
			If true then run tiny3 and use it's output as results. Otherwise the result is tiny2 result. 

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
			self.rootname+'00_psf.fits'
		"""
		if not os.path.isdir(self.dir_out):
			os.makedirs(self.dir_out)

		status1 = call_tinytim.tiny1(dir_code=self.dir_tinytim, fn=self.fp_param, camera=self.camera, detector=self.detector, position=self.position, filter=self.filter, spectrum_form=self.spectrum_form, spectrum_type=self.spectrum_type, diameter=self.diameter, focus=self.focus, rootname=self.rootname) 

		status2 = call_tinytim.tiny2(dir_code=self.dir_tinytim, fn=self.fp_param, rootname=self.rootname) 

		if not run_tiny3:
			os.rename(self.rootname+'00_psf.fits', self.fp_psf)
			status_final = os.path.isfile(self.fp_psf)
			status = np.all([status1, status2, status_final])
			return status

		else:
			status3 = call_tinytim.tiny3(dir_code=self.dir_tinytim, fn=self.fp_param, rootname=self.rootname) 
			os.rename(self.rootname+'00.fits', self.fp_psf)
			status_final = os.path.isfile(self.fp_psf)
			status = np.all([status1, status2, status3, status_final])
			return status


	def read_psf(self):
		"""
		read and return psf np array

		Params
		------
		none

		Return
		------
		psf (np array)
		"""
		return fits.getdata(self.fp_psf)
