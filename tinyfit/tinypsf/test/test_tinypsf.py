import pytest
import os
import numpy as np
import shutil
from astropy.io import fits

from ..tinypsf import tinypsf


dir_testing = 'testing/'
dir_verif = 'data/'

fn = 'j1652_wfc3'

camera = 'wfc3_ir'
position = [563, 561]
filter = 'f160w'
spectrum_form = 'stellar'
spectrum_type = 'f8v'
diameter = 3
focus = -0.5


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


def test_tinypsf_init():

	t = tinypsf(dir_out=dir_testing, fn=fn, camera=camera, filter=filter, position=position, spectrum_form=spectrum_form, spectrum_type=spectrum_type, diameter=diameter, focus=focus, )

	# check that t is a tinypsf
	assert type(t) is tinypsf


def test_tinypsf_make_run_tiny3():

	t = tinypsf(dir_out=dir_testing, fn=fn, camera=camera, filter=filter, position=position, spectrum_form=spectrum_form, spectrum_type=spectrum_type, diameter=diameter, focus=focus, )
	status = t.make_psf(run_tiny3=True)
	assert status

	# check that the file exist
	assert os.path.isfile(t.fp_psf)

	# check that the psf is read as a np array
	psf = t.read_psf()	
	assert type(psf) is np.ndarray

	assert t.fp_psf == dir_testing+fn+'.fits'

	# check that the output is the same
	fn_verif = 'data/j1652_wfc300.fits'
	fn_testing = t.fp_psf
	assert np.all(fits.getdata(fn_testing) == fits.getdata(fn_verif))



def test_tinypsf_make_dont_run_tiny3():

	t = tinypsf(dir_out=dir_testing, fn=fn, camera=camera, filter=filter, position=position, spectrum_form=spectrum_form, spectrum_type=spectrum_type, diameter=diameter, focus=focus, )
	status = t.make_psf(run_tiny3=False)
	assert status

	# check that the file exist
	assert os.path.isfile(t.fp_psf)

	# check that the psf is read as a np array
	psf = t.read_psf()	
	assert type(psf) is np.ndarray

	assert t.fp_psf == dir_testing+fn+'.fits'

	# check that the output is the same
	fn_verif = 'data/j1652_wfc300_psf.fits'
	fn_testing = t.fp_psf
	assert np.all(fits.getdata(fn_testing) == fits.getdata(fn_verif))


