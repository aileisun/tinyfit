import pytest
import os
import shutil
import matplotlib.pyplot as plt

from .. import visual

dir_testing = 'testing/'
dir_data = 'data/'

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


def test_visual_plot():

	v = visual.visual(fn=dir_data+'drz_crop.fits')

	fn_out = dir_testing+'drz_crop.pdf'
	v.plot(fn_out=fn_out, vmin=-0.2, vmax=None, figsize=(3, 3))
	assert os.path.isfile(fn_out)

	fn_out = dir_testing+'drz_crop_cbar.pdf'
	v.plot(fn_out=fn_out, vmin=-0.2, vmax=None, colorbar=True, figsize = (8, 6))
	assert os.path.isfile(fn_out)

	fn_out = dir_testing+'drz_crop_cbar_jet.pdf'
	v.plot(fn_out=fn_out, vmin=-0.2, vmax=None, cmap=plt.cm.jet, colorbar=True, figsize = (8, 6))
	assert os.path.isfile(fn_out)


def test_visual_biplot():

	v = visual.visual(fn=dir_data+'drz_crop.fits')

	fn_out = dir_testing+'drz_crop_biplot_linear.pdf'
	v.biplot(fn=dir_data+'drz_psfsub_crop.fits', stretch='linear', cmap=plt.cm.jet, fn_out=fn_out, colorbar=True)
	assert os.path.isfile(fn_out)


	fn_out = dir_testing+'drz_crop_biplot_log.pdf'
	v.biplot(fn=dir_data+'drz_psfsub_crop.fits', stretch='log', cmap=plt.cm.jet, fn_out=fn_out, colorbar=True)
	assert os.path.isfile(fn_out)


def test_visual_ratioplot():

	v = visual.visual(fn=dir_data+'drz_crop.fits')

	fn_out = dir_testing+'drz_crop_ratioplot_linear.pdf'
	v.ratioplot(fn=dir_data+'drz_psfsub_crop.fits', stretch='linear', cmap=plt.cm.jet, fn_out=fn_out, colorbar=True)
	assert os.path.isfile(fn_out)

