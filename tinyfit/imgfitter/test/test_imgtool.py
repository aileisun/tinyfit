import pytest
import os
import shutil
import numpy as np
from astropy.io import fits

from .. import imgtools

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


def test_imgtools_shift():

	dx, dy = 0.3, 0.3
	subsample = 5
	data = fits.getdata(dir_verif+'j1652_wfc3_sub5.fits')

	data_shift = imgtools.shift(data, dx=dx, dy=dy, subsample=subsample)

	# check flux conservation
	assert np.absolute(data_shift.sum() - data.sum())/data.sum() < 1.e-6

	# check the shift amount is correct
	ny, nx = data.shape
	assert data.max() == data[ny//2, nx//2]

	assert data_shift.max() == data_shift[ny//2+round(dy*subsample), nx//2+round(dx*subsample)]

	# check flux conservation for fractional shift
	data_shift = imgtools.shift(data, dx=0.27990, dy=0.5812, subsample=subsample)
	assert np.absolute(data_shift.sum() - data.sum())/data.sum() < 1.e-6

	# write output file for visual inspection
	hdu = fits.PrimaryHDU(data_shift)
	hdu.writeto(dir_testing+'psf_shifted.fits')	


def test_imgtools_resample():

	nx, ny = 64, 64
	subsample = 5
	data = fits.getdata(dir_verif+'j1652_wfc3_sub5.fits')

	# check that the peak is in the center of the original data
	ny_data, nx_data = data.shape
	assert data.max() == data[ny_data//2, nx_data//2]

	# resample
	data_resamp = imgtools.resample(data, nx, ny, subsample)

	# check flux conservation
	assert np.absolute(data_resamp.sum() - data.sum())/data.sum() < 1.e-6

	# check the peak is still at the center
	assert data_resamp.max() == data_resamp[ny//2, nx//2]

	# check dimension correct
	assert data_resamp.shape == (ny, nx)

	# write output file for visual inspection
	hdu = fits.PrimaryHDU(data_resamp)
	hdu.writeto(dir_testing+'psf_resampled.fits')