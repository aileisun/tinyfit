import pytest
import os
import numpy as np
from astropy.io import fits
import shutil
import json

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


def test_imgfitter_crop():

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

	img_uncrop = f._uncrop(f.img_crop, xc=xstar, yc=ystar)
	assert img_uncrop.data.shape == f.img.data.shape
	assert img_uncrop.data.max() == f.img_crop.data.max()
	assert img_uncrop.data[ystar, xstar] == f.img_crop.data[ny//2, nx//2]
	img_uncrop.writeto(dir_testing+'img_uncrop.fits')


def test_imgfitter_resample_model():

	nx, ny = 64, 64
	dx, dy = 1, 0.8
	scale = 10.
	
	fn = dir_verif+'science_img.fits'
	fn_model = dir_verif+'j1652_wfc3_sub5.fits'

	f = imgfitter(filename=fn, pixsize=0.13, nx=nx, ny=ny)
	f.set_model(filename=fn_model)

	model_resamp = f._get_shifted_scaled_resampled_model(model=f.model_init, dx=dx, dy=dy, scale=scale)

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

	# no gaussian smoothing
	status = f.fit(x=xstar, y=ystar)

	chisq_nogauss = f.result.chisq
	assert status
	assert f.result.chisq > 0.
	assert f.result.chisq < 2000.

	assert f.img_crop_bestfit.data.shape == (ny, nx)
	assert f.img_crop_residual.data.shape == (ny, nx)

	f.img_crop.writeto(dir_testing+'img_crop.fits')
	f.img_crop_residual.writeto(dir_testing+'residual_crop.fits')
	assert os.path.isfile(dir_testing+'residual_crop.fits')

	f.result.save(dir_testing+'result.json')
	assert os.path.isfile(dir_testing+'result.json')

	f.img_bestfit.writeto(dir_testing+'img_bestfit.fits')
	assert os.path.isfile(dir_testing+'img_bestfit.fits')
	assert f.img_bestfit.data.shape == fits.getdata(fn).shape


def test_imgfitter_fit_gaussiansmoothing():

	xstar, ystar = 512, 511
	nx, ny = 64, 64

	fn = dir_verif+'science_img.fits'
	fn_model = dir_verif+'j1652_wfc3_sub5.fits'

	f = imgfitter(filename=fn, pixsize=0.13, nx=nx, ny=ny, )
	f.set_model(filename=fn_model)

	# no gaussian smoothing
	status = f.fit(x=xstar, y=ystar)

	chisq_nogauss = f.result.chisq
	assert status
	assert f.result.chisq > 0.

	assert f.img_crop_bestfit.data.shape == (ny, nx)
	assert f.img_crop_residual.data.shape == (ny, nx)

	f.img_crop_residual.writeto(dir_testing+'residual.fits')
	assert os.path.isfile(dir_testing+'residual.fits')

	# with gaussian smoothing
	freeparams = ['dx', 'dy', 'scale', 'sigma']
	status = f.fit(x=xstar, y=ystar, freeparams=freeparams)
	chisq_gauss = f.result.chisq
	f.img_crop_bestfit.writeto(dir_testing+'sigma_bestfit.fits')
	f.img_crop_residual.writeto(dir_testing+'sigma_residual.fits')

	# assert smoothing is better
	assert chisq_gauss < chisq_nogauss
	assert f.result.params.sigma > 0.


def test_imgfitter_hyperfit():

	# img
	pixsize_img = 0.13
	fn_img = dir_verif+'science_img.fits'
	nx, ny = 64, 64

	# # psf
	# xstar, ystar = 512, 511
	# camera = 'wfc3_ir'
	# filter = 'f160w'
	# spectrum_form = 'stellar'
	# spectrum_type = 'k7v'
	# diameter = 6
	# focus = -0.5
	# subsample = 5
	# fn_psf = 'j1652_wfc3_star1'

	# tpsf = tinypsf(camera=camera, filter=filter, position=[xstar, ystar], spectrum_form=spectrum_form, spectrum_type=spectrum_type, diameter=diameter, focus=focus, subsample=subsample, fn=fn_psf, dir_out=dir_testing)

	tpsf = get_standard_tpsf()

	f = imgfitter(filename=fn_img, pixsize=pixsize_img, nx=nx, ny=ny)
	f.set_model(tinypsf=tpsf)

	# normal fit
	f.fit(x=xstar, y=ystar)
	chisq_normal = f.result.chisq
	focus_normal = f.hypermodel_init.focus
	f.img_crop_residual.writeto(dir_testing+'residual.fits')

	# hyper fit
	f.hyperfit(x=xstar, y=ystar, freehyperparams=['focus'])
	chisq_hyper = f.result.chisq
	focus_hyper = f.result.hyperparams.focus
	f.img_crop_residual.writeto(dir_testing+'hyper_residual.fits')

	assert chisq_hyper < chisq_normal
	assert focus_normal != focus_hyper

	# hyper fit with smoothing
	freeparams = ['dx', 'dy', 'scale', 'sigma']
	f.hyperfit(x=xstar, y=ystar, freeparams=freeparams, freehyperparams=['focus'])
	chisq_hypersmooth = f.result.chisq
	f.img_crop_residual.writeto(dir_testing+'hypersmooth_residual.fits')

	# assert chisq_hypersmooth < chisq_hyper
	assert f.result.params.sigma > 0.
	assert chisq_hypersmooth != chisq_hyper


def test_imgfitter_charge_diffusion():

	xstar, ystar = 512, 511
	nx, ny = 64, 64
	fn = dir_verif+'science_img.fits'
	fn_psf = 'j1652_wfc3_sub5'

	t = tinypsf(dir_out=dir_testing, fn=fn_psf, subsample=5)
	# t.load_psf()

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

	f.result._set_params_from_OptimizeResult(r, names=['dx', 'dy', 'scale'])
	assert f.result.params.dx == 1
	assert f.result.success == r.success
	assert f.result.chisq == r.fun
	assert f.result.ishyper is True

	f.result._set_hyperparams_from_OptimizeResult(rh, names=['focus'])
	assert f.result.hyperparams.focus == 0.5
	assert f.result.hyperparams._asdict()['focus'] == 0.5
	assert f.result.chisq == rh.fun
	assert f.result.ishyper is True



def test_imgfitter_save_result():

	xstar, ystar = 512, 511
	nx, ny = 64, 64

	fn = dir_verif+'science_img.fits'
	fn_model = dir_verif+'j1652_wfc3_sub5.fits'

	f = imgfitter(filename=fn, pixsize=0.13, nx=nx, ny=ny, )
	f.set_model(filename=fn_model)

	# no gaussian smoothing
	status = f.fit(x=xstar, y=ystar)
	assert status
	chisq = f.result.chisq
	params = f.result.params
	hyperparams = f.result.hyperparams

	fn_result = dir_testing+'result.json'
	f.result.save(fn_result)

	assert os.path.isfile(fn_result)

	with open(fn_result, 'r') as file:
		res = json.load(file)

	assert res['params']['dx'] != 0.
	assert res['hyperparams']  == {}
	assert res['chisq'] > 0.
	assert res['success'] == True


	f2 = imgfitter(filename=fn, pixsize=0.13, nx=nx, ny=ny, )
	assert f2.result.chisq is None
	f2.result.load(fn_result)
	assert f2.result.chisq == chisq

	assert dict(f2.result.params._asdict()) == dict(params._asdict())
	assert dict(f2.result.hyperparams._asdict()) == (hyperparams._asdict())



def test_imgfitter_fit_model_maxlim():
	""" limit the max of the model to be smaller than, say, the maximum of the data image. 
	"""

	xstar, ystar = 512, 511
	nx, ny = 64, 64

	fn = dir_verif+'science_img.fits'
	fn_model = dir_verif+'j1652_wfc3_sub5.fits'

	f = imgfitter(filename=fn, pixsize=0.13, nx=nx, ny=ny, )
	f.set_model(filename=fn_model)

	freeparams = ['dx', 'dy', 'scale', 'sigma']
	status = f.fit(x=xstar, y=ystar, freeparams=freeparams)
	assert status
	model_max = f.img_crop_bestfit.data.max()
	data_max = f.img_crop.data.max()

	f.img_crop.writeto(dir_testing+'img.fits')
	f.img_crop_residual.writeto(dir_testing+'residual.fits')
	f.img_bestfit.writeto(dir_testing+'img_bestfit.fits')
	f.result.save(dir_testing+'result.json')

	# with model_maxlim
	freeparams = ['dx', 'dy', 'scale', 'sigma']
	status = f.fit(x=xstar, y=ystar, freeparams=freeparams, model_maxlim='data')
	assert status

	assert f.img_crop_bestfit.data.max() < model_max
	assert f.img_crop_bestfit.data.max() <= data_max

	f.img_crop_residual.writeto(dir_testing+'residual_maxlim.fits')
	f.img_bestfit.writeto(dir_testing+'img_bestfit_maxlim.fits')
	f.result.save(dir_testing+'result_maxlim.json')

	# hyper
	# xstar, ystar = 512, 511
	# camera = 'wfc3_ir'
	# filter = 'f160w'
	# spectrum_form = 'stellar'
	# spectrum_type = 'k7v'
	# diameter = 6
	# focus = -0.5
	# subsample = 5
	# fn_psf = 'j1652_wfc3_star1'

	# tpsf = tinypsf(camera=camera, filter=filter, position=[xstar, ystar], spectrum_form=spectrum_form, spectrum_type=spectrum_type, diameter=diameter, focus=focus, subsample=subsample, fn=fn_psf, dir_out=dir_testing)

	tpsf = get_standard_tpsf()

	f.set_model(tinypsf=tpsf)
	status = f.hyperfit(x=xstar, y=ystar, freeparams=freeparams, freehyperparams=['focus'], model_maxlim='data')
	assert status

	assert f.img_crop_bestfit.data.max() < model_max
	assert f.img_crop_bestfit.data.max() <= data_max

	f.img_crop_residual.writeto(dir_testing+'residual_maxlim_hyper.fits')
	f.img_bestfit.writeto(dir_testing+'img_bestfit_maxlim_hyper.fits')
	f.result.save(dir_testing+'result_maxlim_hyper.json')



def test_imgfitter_fit_neg_penal():
	""" penalize the negtive residuals by a factor of 'neg_penal' in the chisq calculation. 
	"""

	xstar, ystar = 512, 511
	nx, ny = 64, 64

	fn = dir_verif+'science_img.fits'
	fn_model = dir_verif+'j1652_wfc3_sub5.fits'

	f = imgfitter(filename=fn, pixsize=0.13, nx=nx, ny=ny, )
	f.set_model(filename=fn_model)

	freeparams = ['dx', 'dy', 'scale', 'sigma']
	status = f.fit(x=xstar, y=ystar, freeparams=freeparams)
	assert status
	model_max = f.img_crop_bestfit.data.max()
	data_max = f.img_crop.data.max()

	f.img_crop.writeto(dir_testing+'img.fits')
	f.img_crop_residual.writeto(dir_testing+'residual.fits')
	f.img_crop_bestfit.writeto(dir_testing+'img_bestfit.fits')
	f.result.save(dir_testing+'result.json')

	# with neg_penal
	freeparams = ['dx', 'dy', 'scale', 'sigma']
	status = f.fit(x=xstar, y=ystar, freeparams=freeparams, neg_penal=3.)
	assert status

	assert f.img_crop_bestfit.data.max() < model_max
	assert f.img_crop_bestfit.data.max() <= data_max

	f.img_crop_residual.writeto(dir_testing+'residual_negpenal.fits')
	f.img_crop_bestfit.writeto(dir_testing+'img_bestfit_negpenal.fits')
	f.result.save(dir_testing+'result_negpenal.json')


	# with hyperfit
	tpsf = get_standard_tpsf()

	f.set_model(tinypsf=tpsf)

	status = f.hyperfit(x=xstar, y=ystar, freeparams=freeparams, freehyperparams=['focus'], neg_penal=3.)
	assert status

	assert f.img_crop_bestfit.data.max() < model_max
	assert f.img_crop_bestfit.data.max() <= data_max

	f.img_crop_residual.writeto(dir_testing+'residual_negpenal_hyper.fits')
	f.img_crop_bestfit.writeto(dir_testing+'img_bestfit_negpenal_hyper.fits')
	f.result.save(dir_testing+'result_negpenal_hyper.json')


def get_standard_tpsf():
	xstar, ystar = 512, 511
	camera = 'wfc3_ir'
	filter = 'f160w'
	spectrum_form = 'stellar'
	spectrum_type = 'k7v'
	diameter = 6
	focus = -0.5
	subsample = 5
	fn_psf = 'j1652_wfc3_star1'

	return tinypsf(camera=camera, filter=filter, position=[xstar, ystar], spectrum_form=spectrum_form, spectrum_type=spectrum_type, diameter=diameter, focus=focus, subsample=subsample, fn=fn_psf, dir_out=dir_testing)


def test_imgfitter_fit_param_range():

	xstar, ystar = 512, 511
	nx, ny = 64, 64

	fn = dir_verif+'science_img.fits'
	fn_model = dir_verif+'j1652_wfc3_sub5.fits'

	f = imgfitter(filename=fn, pixsize=0.13, nx=nx, ny=ny, )
	f.set_model(filename=fn_model)

	# no gaussian smoothing
	freeparams = ['dx', 'dy', 'scale', 'sigma']

	status = f.fit(x=xstar, y=ystar, freeparams=freeparams)
	chisq_norange = f.result.chisq

	f.img_crop.writeto(dir_testing+'img_crop.fits')
	f.img_crop_residual.writeto(dir_testing+'residual_crop.fits')
	f.result.save(dir_testing+'result.json')

	params_range = {'dx':(-2., -1.), 'dy':(-2, 0.), }
	status = f.fit(x=xstar, y=ystar, freeparams=freeparams, params_range=params_range)

	f.img_crop_residual.writeto(dir_testing+'residual_crop_range.fits')
	f.result.save(dir_testing+'result_range.json')

	assert status
	assert f.result.chisq > 0.
	# assert f.result.chisq > chisq_norange
	assert f.result.params.dx > params_range['dx'][0]
	assert f.result.params.dx < params_range['dx'][1]
	assert f.result.params.dy > params_range['dy'][0]
	assert f.result.params.dy < params_range['dy'][1]


	# hyper

	tpsf = get_standard_tpsf()
	f.set_model(tinypsf=tpsf)
	status = f.hyperfit(x=xstar, y=ystar, freeparams=freeparams, freehyperparams=['focus'], params_range=params_range)

	f.img_crop_residual.writeto(dir_testing+'residual_crop_hyper_range.fits')
	f.result.save(dir_testing+'result_hyper_range.json')


	assert status
	assert f.result.chisq > 0.
	# assert f.result.chisq > chisq_norange
	assert f.result.params.dx > params_range['dx'][0]
	assert f.result.params.dx < params_range['dx'][1]
	assert f.result.params.dy > params_range['dy'][0]
	assert f.result.params.dy < params_range['dy'][1]
