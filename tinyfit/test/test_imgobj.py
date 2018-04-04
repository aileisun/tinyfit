import pytest
from astropy.io import fits
import numpy as np


from .. import imgobj

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