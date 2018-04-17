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
subsample = 1


@pytest.fixture(scope="module", autouse=True)
def setUp_tearDown():
	""" rm ./testing/ before and after testing"""

	# setup
	if os.path.isdir(dir_testing):
		shutil.rmtree(dir_testing)
	os.mkdir(dir_testing)

	yield
	# tear down
	if os.path.isdir(dir_testing):
		shutil.rmtree(dir_testing)


def test_tinypsf_init():

	t = tinypsf(dir_out=dir_testing, fn=fn, camera=camera, filter=filter, position=position, spectrum_form=spectrum_form, spectrum_type=spectrum_type, diameter=diameter, focus=focus, subsample=subsample)

	# check that t is a tinypsf
	assert type(t) is tinypsf


def test_tinypsf_make_psf():

	t = tinypsf(dir_out=dir_testing, fn=fn, camera=camera, filter=filter, position=position, spectrum_form=spectrum_form, spectrum_type=spectrum_type, diameter=diameter, focus=focus, subsample=subsample)
	status = t.make_psf()
	assert status

	# check that the file exist
	assert os.path.isfile(t.fp_psf)

	# check that the output is the same
	fn_verif = 'data/j1652_wfc300.fits'
	fn_testing = t.fp_psf
	assert np.all(fits.getdata(fn_testing) == fits.getdata(fn_verif))

	# check that the psf is read as a np array
	psf = t.get_psf()	
	assert type(psf) is np.ndarray
	assert t.fp_psf == dir_testing+fn+'.fits'

	# check that psf meta data is loaded properly
	status = t.load_psf()
	assert status
	assert type(t.psf.data) is np.ndarray
	assert t.psf.subsample == 1
	assert t.psf.pixsize == 0.130
	assert t.psf.require_diffusion == False
	assert t.psf.diffusion_kernel is None
	assert 'PIXSCALE' in t.psf.header



def test_tinypsf_make_psf_powerlaw_nu():

	fn = 'j1652_wfc3_powerlaw_nu'
	t = tinypsf(dir_out=dir_testing, fn=fn, camera=camera, filter=filter, position=position, spectrum_form='powerlaw_nu', spectrum_type=1.5, diameter=diameter, focus=focus, subsample=subsample)
	status = t.make_psf()
	assert status

	# check that the file exist
	assert os.path.isfile(t.fp_psf)

	# check that the output is the same
	fn_verif = 'data/j1652_wfc3_powerlaw_nu00.fits'
	fn_testing = t.fp_psf
	assert np.all(fits.getdata(fn_testing) == fits.getdata(fn_verif))



def test_tinypsf_make_psf_subsample5():

	t = tinypsf(dir_out=dir_testing, fn=fn, camera=camera, filter=filter, position=position, spectrum_form=spectrum_form, spectrum_type=spectrum_type, diameter=diameter, focus=focus, subsample=5)
	status = t.make_psf()
	assert status

	# check that the file exist
	assert os.path.isfile(t.fp_psf)

	# check that the output is the same as the verification data
	fn_verif = 'data/j1652_wfc3_sub500.fits'
	fn_testing = t.fp_psf
	assert np.all(fits.getdata(fn_testing) == fits.getdata(fn_verif))

	# check that the psf is read as a np array
	psf = t.get_psf()	
	assert type(psf) is np.ndarray
	assert t.fp_psf == dir_testing+fn+'.fits'

	# check that psf meta data is loaded properly
	status = t.load_psf()
	assert status
	assert type(t.psf.data) is np.ndarray
	assert t.psf.subsample == 5
	assert t.psf.pixsize == 0.0260
	assert t.psf.require_diffusion == True
	verif_diffusion_kernel = np.array([[0.000700, 0.025005, 0.000700, ],
										[0.025005, 0.897179, 0.025005, ],
										[0.000700, 0.025005, 0.000700, ],
										])
	assert np.all(t.psf.diffusion_kernel == verif_diffusion_kernel)
	assert 'PIXSCALE' in t.psf.header
	