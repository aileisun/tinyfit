import pytest
import os
import numpy as np
from astropy.io import fits
import shutil

from ..imgfitter import imgfitter
from ...tinypsf import tinypsf
from ... import imgobj

dir_testing = 'testing/'
dir_verif = 'data/'


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


def test_imgfitter_initiate():

	fn = dir_verif+'science_img.fits'
	f = imgfitter(filename=fn, pixsize=0.13)

	assert type(f) is imgfitter
	assert f.img.data.shape == (1014, 1014)
	assert f.img.nx == 1014
	assert f.img.xc == 1014//2
	assert f.img.pixsize == 0.13


def test_imgfitter_cropimg():

	xstar, ystar = 512, 511
	nx, ny = 64, 64

	fn = dir_verif+'science_img.fits'
	f = imgfitter(filename=fn, pixsize=0.13, nx=nx, ny=ny)

	f._cropimg(xc=xstar, yc=ystar)

	assert f._cropxc == xstar
	assert f._cropyc == ystar
	assert f.img_crop.data.shape == (ny, nx)
	assert f.img_crop.nx == nx
	assert f.img_crop.pixsize == 0.13
	assert f.img_crop.data[ny//2, nx//2] == f.img.data[ystar, xstar]

	xstar_test, ystar_test = f._cropxy_to_xy(x=nx//2, y=ny//2)
	assert (xstar_test, ystar_test) == (xstar, ystar)

	xstar_test2, ystar_test2 = f._cropxy_to_xy(x=nx//2+1, y=ny//2+2)
	assert (xstar_test2, ystar_test2) == (xstar+1, ystar+2)


def test_imgfitter_resample_model():

	nx, ny = 64, 64
	dx, dy = 1, 0.8
	scale = 10.
	
	fn = dir_verif+'science_img.fits'
	fn_model = dir_verif+'j1652_wfc3_sub5.fits'

	f = imgfitter(filename=fn, pixsize=0.13, nx=nx, ny=ny)
	f.set_model(filename=fn_model)

	model_resamp = f._get_shifted_resampled_model(model=f.model_init, dx=dx, dy=dy, scale=scale)

	assert np.absolute(np.sum(model_resamp.data)/scale - np.sum(f.model_init.data))/np.sum(f.model_init.data) < 1.e-5

	assert model_resamp.nx == nx
	assert model_resamp.data[ny//2+round(dy), nx//2+round(dx)] == model_resamp.data.max()


def test_imgfitter_fit():

	xstar, ystar = 512, 511
	nx, ny = 64, 64

	fn = dir_verif+'science_img.fits'
	fn_model = dir_verif+'j1652_wfc3_sub5.fits'

	f = imgfitter(filename=fn, pixsize=0.13, nx=nx, ny=ny, )
	f.set_model(filename=fn_model)

	status = f.fit(x=xstar, y=ystar)

	assert status
	assert f.result.chisq > 0.

	assert f.img_crop_bestfit.data.shape == (ny, nx)
	assert f.img_crop_residual.data.shape == (ny, nx)

	f.img_crop_residual.writeto(dir_testing+'residual.fits')
	assert os.path.isfile(dir_testing+'residual.fits')


def test_imgfitter_charge_diffusion():

	xstar, ystar = 512, 511
	nx, ny = 64, 64
	fn = dir_verif+'science_img.fits'
	fn_psf = 'j1652_wfc3_sub5'

	t = tinypsf(dir_out=dir_verif, fn=fn_psf, subsample=5)
	t.load_psf()

	f = imgfitter(filename=fn, pixsize=0.13, nx=nx, ny=ny, charge_diffusion=True)
	f.set_model(tinypsf=t)

	# no diffusion
	status = f.fit(x=xstar, y=ystar, charge_diffusion=False)
	chisq_nodiffusion =  f.result.chisq
	f.img_crop_residual.writeto(dir_testing+'residual_withno_diffusion.fits')
	assert f.charge_diffusion == False

	# diffusion
	status = f.fit(x=xstar, y=ystar, charge_diffusion=True)
	chisq_diffusion = f.result.chisq
	f.img_crop_residual.writeto(dir_testing+'residual_with_diffusion.fits')
	assert f.charge_diffusion == True

	# comparison
	assert chisq_diffusion < chisq_nodiffusion


def test_imgfitter_hyperfit():

	# img
	pixsize_img = 0.13
	fn_img = dir_verif+'science_img.fits'
	nx, ny = 64, 64

	# psf
	xstar, ystar = 512, 511
	camera = 'wfc3_ir'
	filter = 'f160w'
	spectrum_form = 'stellar'
	spectrum_type = 'k7v'
	diameter = 6
	focus = -0.5
	subsample = 5
	fn_psf = 'j1652_wfc3_star1'

	tpsf = tinypsf(camera=camera, filter=filter, position=[xstar, ystar], spectrum_form=spectrum_form, spectrum_type=spectrum_type, diameter=diameter, focus=focus, subsample=subsample, fn=fn_psf, dir_out=dir_testing)

	f = imgfitter(filename=fn_img, pixsize=pixsize_img, nx=nx, ny=ny)
	f.set_model(tinypsf=tpsf)

	# normal fit
	f.fit(x=xstar, y=ystar)
	chisq_normal = f.result.chisq
	focus_normal = f.hypermodel_init.focus

	# hyper fit
	f.hyperfit(x=xstar, y=ystar, hyperparams=['focus'])
	chisq_hyper = f.result.chisq
	focus_hyper = f.result.hyperparams.focus

	assert chisq_hyper < chisq_normal
	assert focus_normal != focus_hyper

	f.img_crop_residual.writeto(dir_testing+'residual.fits')


def test_imgfitter_result():

	class OptimizeResult():
		pass
	r = OptimizeResult()
	r.x = np.array([1, 2, 3])
	r.success = True
	r.fun = 12345.

	rh = OptimizeResult()
	rh.x = np.array([0.5])
	rh.success = True
	rh.fun = 12321.

	xstar, ystar = 512, 511
	nx, ny = 64, 64

	fn = dir_verif+'science_img.fits'
	fn_model = dir_verif+'j1652_wfc3_sub5.fits'

	f = imgfitter(filename=fn, pixsize=0.13, nx=nx, ny=ny, )
	f.set_model(filename=fn_model)
	f.result._set_params([2, 3, 4], names=['dx', 'dy', 'scale'])
	assert f.result.params.dx == 2
	assert f.result.params.dy == 3
	assert f.result.ishyper is False

	f.result._set_hyperparams([0.9], names=['focus'])
	assert f.result.params.dx == 2
	assert f.result.hyperparams.focus == 0.9
	assert f.result.ishyper is True

	f.result._set_from_OptimizeResult(r, mode='params', names=['dx', 'dy', 'scale'])
	assert f.result.params.dx == 1
	assert f.result.success == r.success
	assert f.result.chisq == r.fun
	assert f.result.ishyper is True

	f.result._set_from_OptimizeResult(rh, mode='hyperparams', names=['focus'])
	assert f.result.hyperparams.focus == 0.5
	assert f.result.hyperparams._asdict()['focus'] == 0.5
	assert f.result.chisq == rh.fun
	assert f.result.ishyper is True
