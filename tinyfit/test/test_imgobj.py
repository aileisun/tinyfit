import pytest
from astropy.io import fits
import numpy as np
import os
import shutil

from .. import imgobj

dir_testing = 'testing/'


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


def test_imgobj_init_file():

	fn = '../imgfitter/test/data/science_img.fits'
	img = imgobj(filename=fn, pixsize=0.13)

	assert img.pixsize == 0.13
	assert img.nx == 1014
	assert img.data.shape == (1014, 1014)


def test_imgobj_init_imagearray():

	fn = '../tinypsf/test/data/j1652_wfc300.fits'
	data = fits.getdata(fn)

	img = imgobj(data=data, pixsize=0.13)

	assert img.pixsize == 0.13
	assert img.nx == 27
	assert img.data.shape == (27, 27)


def test_imgobj_saveto():

	fn = '../tinypsf/test/data/j1652_wfc300.fits'
	data = fits.getdata(fn)

	img = imgobj(data=data, pixsize=0.13)

	img.writeto(dir_testing+'test.fits')

	img_new = imgobj(filename=dir_testing+'test.fits')

	assert np.all(img.data == img_new.data)
	assert img.pixsize == img_new.pixsize