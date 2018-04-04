#!/usr/bin/env python
"""
Test call_tinytim tiny1
"""

import pytest
import os
import numpy as np
import filecmp
import shutil
from astropy.io import fits

from .. import call_tinytim


dir_verif = 'data/'
dir_testing = 'testing/'
dir_code = '/Users/aisun/Documents/astro/algorithm/tinytim/tinytim-7.5/'


@pytest.fixture(scope="module", autouse=True)
def setUp_tearDown():
	""" rm ./testing/ before and after testing"""

	# setup
	if os.path.isdir(dir_testing):
		shutil.rmtree(dir_testing)
	os.mkdir(dir_testing)

	# yield
	# # tear down
	# if os.path.isdir(dir_testing):
	# 	shutil.rmtree(dir_testing)


def test_calltinytim_tiny1_wfc3():
	"""
	# manual tiny1 setup: 
	> tiny1 j1652_wfc3.param
	Camera: 
		23) WFC3 IR channel
	Position: 
		563 561 
		(the img x, y position as read from the `id9ya9v8q_ima.fits` file). 
	Select filter passband: 
		f160w
	object spectrum:  **Random guess — to be improved**
		1) Select a spectrum from list. 
		10   F8V      0.62   0.48   0.37   0.68   0.98   1.25
	What diameter should your PSF be (in arcseconds)? : 
		3
	Focus, secondary mirror despace? [microns]: 
		-0.5 
	Rootname of PSF image files (no extension) :
		j1652_wfc3
	"""
	# tiny1 params
	name = 'j1652_wfc3'
	fn = dir_testing+name+'.param'
	camera = 'wfc3_ir'
	position = [563, 561]
	filter = 'f160w'
	spectrum_form = 'stellar'
	spectrum_type = 'f8v'
	diameter = 3
	focus = -0.5
	rootname = dir_testing+name

	status = call_tinytim.tiny1(dir_code=dir_code, fn=fn, camera=camera, position=position, filter=filter, spectrum_form=spectrum_form, spectrum_type=spectrum_type, diameter=diameter, focus=focus, rootname=rootname)

	assert status
	assert os.path.isfile(fn)
	assert filecmp.cmp(dir_testing+name+'.param', dir_verif+name+'.param')


def test_calltinytim_tiny2_wfc3():
	"""
	check that correct file is produced with the correct content. 
	"""
	test_calltinytim_tiny1_wfc3()

	name = 'j1652_wfc3'
	fn = dir_testing+name+'.param'
	rootname = dir_testing+name

	status = call_tinytim.tiny2(dir_code=dir_code, fn=fn, rootname=rootname)

	# check files have the right content
	assert status
	for suffix in ['.tt3']: 
		assert os.path.isfile(dir_testing+name+suffix)
		assert filecmp.cmp(dir_testing+name+suffix, dir_verif+name+suffix)

	suffix = '00_psf.fits'
	fn_verif = dir_verif+name+suffix
	fn_testing = dir_testing+name+suffix
	data_verif = fits.getdata(fn_verif)
	data_testing = fits.getdata(fn_testing)
	assert np.all(data_testing == data_verif)


def test_calltinytim_tiny3_wfc3():
	"""
	check that correct file is produced with the correct content. 
	"""
	test_calltinytim_tiny1_wfc3()
	test_calltinytim_tiny2_wfc3()

	name = 'j1652_wfc3'
	fn = dir_testing+name+'.param'
	rootname = dir_testing+name

	status = call_tinytim.tiny3(dir_code=dir_code, fn=fn, rootname=rootname, subsample=1)

	assert status
	suffix = '00.fits'
	fn_verif = dir_verif+name+suffix
	fn_testing = dir_testing+name+suffix
	data_verif = fits.getdata(fn_verif)
	data_testing = fits.getdata(fn_testing)
	assert np.all(data_testing == data_verif)



def test_calltinytim_tiny3_wfc3_subsample5():
	"""
	check that correct file is produced with the correct content. 

	# to produce the test file
	> tiny1 j1652_wfc3_sub5.param

		Camera: 
			23) WFC3 IR channel
		Position: 
			563 561 
			(the img x, y position as read from the `id9ya9v8q_ima.fits` file). 
		Select filter passband: 
			f160w
		object spectrum:  **Random guess — to be improved**
			1) Select a spectrum from list. 
			10   F8V      0.62   0.48   0.37   0.68   0.98   1.25
		What diameter should your PSF be (in arcseconds)? : 
			3
		Focus, secondary mirror despace? [microns]: 
			-0.5 
		Rootname of PSF image files (no extension) :
			j1652_wfc3_sub5


	> tiny2 j1652_wfc3_sub5.param

	> tiny3 j1652_wfc3_sub5.param sub=5

	#   NOTE : Subsampled, so not convolving with charge diffusion kernel.

	# output files

	j1652_wfc3_sub5.param
	j1652_wfc3_sub5.tt3
	j1652_wfc3_sub500_psf.fits
	j1652_wfc3_sub500.fits
	"""
	test_calltinytim_tiny1_wfc3()
	test_calltinytim_tiny2_wfc3()

	name = 'j1652_wfc3'
	fn = dir_testing+name+'.param'
	rootname = dir_testing+name

	status = call_tinytim.tiny3(dir_code=dir_code, fn=fn, rootname=rootname, subsample=5)

	assert status
	suffix = '00.fits'
	fn_verif = dir_verif+'j1652_wfc3_sub500.fits'
	fn_testing = dir_testing+name+suffix
	data_verif = fits.getdata(fn_verif)
	data_testing = fits.getdata(fn_testing)
	assert np.all(data_testing == data_verif)

def test_calltinytim_tiny1_acs():
	"""
	# manual tiny1 setup: 
	Cameras
	Choice : 15
	Enter detector (1 or 2) : 1
	Position : 2100 1025
	Filter : f814w

	Choose form of object spectrum :
	    1) Select a spectrum from list
	    2) Blackbody
	    3) Power law : F(nu) = nu^i 
	    4) Power law : F(lambda) = lambda^i 
	    5) Read user-provided spectrum from ASCII table
	Choice : 5
	Enter name of spectrum file : data/spectrum.txt
	What diameter should your PSF be (in arcseconds)? : 5
	Focus, secondary mirror despace? [microns]: 2.0
	Rootname of PSF image files (no extension) : j1652_acs
	"""

	# tiny1 params
	name = 'j1652_acs'
	fn = dir_testing+name+'.param'
	camera = 'acs_widefield'
	detector = 1
	position = [2100, 1025]
	filter = 'f814w'
	spectrum_form = 'user'
	spectrum_type = dir_verif+'spectrum.txt'
	diameter = 5
	focus = 2.
	rootname = dir_testing+name

	status = call_tinytim.tiny1(dir_code=dir_code, fn=fn, camera=camera, detector=detector, position=position, filter=filter, spectrum_form=spectrum_form, spectrum_type=spectrum_type, diameter=diameter, focus=focus, rootname=rootname)

	assert status
	assert os.path.isfile(fn)
	assert filecmp.cmp(dir_testing+name+'.param', dir_verif+name+'.param')



