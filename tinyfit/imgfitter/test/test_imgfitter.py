import pytest
import os
import numpy as np
from astropy.io import fits
import shutil

from ..imgfitter import imgfitter

dir_testing = 'testing/'
dir_verif = 'data/'

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


def test_imgfitter_initiate():

	fn = dir_verif+'science_img.fits'
	f = imgfitter(filename=fn, pixsize=0.13)

	assert type(f) is imgfitter
	assert f.img.data.shape == (1014, 1014)
	assert f.img.nx == 1014
	assert f.img.xc == 1014//2
	assert f.img.pixsize == 0.13


def test_imgfitter_crop():

	xstar, ystar = 512, 511
	nx, ny = 64, 64

	fn = dir_verif+'science_img.fits'
	f = imgfitter(filename=fn)

	f._crop(xc=xstar, yc=ystar, nx=nx, ny=ny)

	assert f._cropxc == xstar
	assert f._cropyc == ystar
	assert f.img_crop.data.shape == (nx, ny)
	assert f.img_crop.nx == ny
	assert f.img_crop.pixsize == 0.13

	xstar_test, ystar_test = f._cropxy2imgxy(x=nx//2, y=ny//2)
	assert (xstar_test, ystar_test) == (xstar, ystar)

	xstar_test2, ystar_test2 = f._cropxy2imgxy(x=nx//2+1, y=ny//2+2)
	assert (xstar_test2, ystar_test2) == (xstar+1, ystar+2)

