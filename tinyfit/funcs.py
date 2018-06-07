"""
operational functions to be called iteratively by batch 
"""

import os
import shutil
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.io import fits
from astropy import wcs
from astropy import coordinates as ac
from astropy import units as u
import astropy.table as at
import photutils as pu

from drizzlepac import astrodrizzle as adz

import tinyfit

def funcdrz_aperture_photometry_on_crop(drz, obs, tar, source='qso', fp_image='drz_crop.fits', radii=([1., 2., 3., 4., ])*u.arcsec): 
	""" measure aperture photometry of images
	"""
	s = drz.sources[source]

	# wcs from drz.fits
	w = wcs.WCS(fits.getheader(drz.directory+'drz.fits', 1))

	# read crop
	fp = s.directory+fp_image
	fp_root = os.path.splitext(fp)[0]
	data = fits.getdata(fp)

	# define aperture
	pos_sky = ac.SkyCoord(s.ra, s.dec, frame='icrs', unit='deg')
	apt_sky = [pu.SkyCircularAperture(pos_sky, r=r) for r in radii]
	apt_pix = [apt_sky.to_pixel(wcs=w) for apt_sky in apt_sky]

	# transform from drz.fits pix coordinate to crop pix coordinate
	pos_pix_crop = apt_pix[0].positions[0] - np.array([s.x, s.y]) + np.array(data.shape[::-1])//2

	apertures_pix_crop = copy.copy(apt_pix)
	for aperture in apertures_pix_crop:
		aperture.positions = np.array([pos_pix_crop])

	phot = at.Table(pu.aperture_photometry(data, apertures_pix_crop))
	phot['xcenter'].unit = None
	phot['ycenter'].unit = None

	# reorganize phot table
	phot.remove_column('id')
	phot.rename_column('xcenter', 'xc_drzcrop')
	phot.rename_column('ycenter', 'yc_drzcrop')
	phot['xc_drz'] = apt_pix[0].positions[0][0]
	phot['yc_drz'] = apt_pix[0].positions[0][1]
	phot['ra'] = s.ra
	phot['dec'] = s.dec
	phot['fn_image'] = fp_image
	cols = ['fn_image', 'ra', 'dec', 'xc_drz', 'yc_drz', 'xc_drzcrop', 'yc_drzcrop', ]+['aperture_sum_'+str(i) for i in range(len(radii))]

	phot = phot[cols]

	# write photometric results
	phot.write(fp_root+'_aphot.csv', overwrite=True)

	# write radii csv
	radii_df = pd.DataFrame({'tag': ['aperture_sum_'+str(i) for i in range(len(radii))], 
							'radii': [str(radii[i]) for i in range(len(radii))]})
	radii_df = radii_df[['tag','radii']]
	radii_df.to_csv('radii.csv', index=False)

	# visual
	v = tinyfit.visual.visual(fp)
	v.plot(fn_out=fp_root+'_aphot.pdf', colorbar=True)
	for a in apertures_pix_crop:
		a.plot()
	plt.savefig(fp_root+'_aphot.pdf')
	plt.close()


def funcflt_stealresult(flt, drz, obs, tar, dir_run_from='../run_oldcentriod/', source='star0'):
	""" copy 'result.json' of star0 from other runs so that one need not rerun focus again """
	if not os.path.isdir(flt.directory+source+'/'):
		os.mkdir(flt.directory+source+'/')

	fp_tail = flt.directory+source+'/result.json'

	fp_result_from = dir_run_from+fp_tail
	fp_result_to = fp_tail

	print('copying '+fp_result_to)

	shutil.copyfile(fp_result_from, fp_result_to)


def funcflt_psffit(flt, drz, obs, tar, source='qso', freeparams=['dx', 'dy', 'scale', 'sigma'], params_range={}, neg_penal=1., refocus=False, source_focus='star0', nx=64, ny=64, pixsize=0.13): 
	""" fitting psf to source for each flt to produce flt/source/flt_psf.fits

	Note:
		If refocus is True, focus is determined from hyperfit to source_focus (e.g., 'star0') then fixed and applied to source. Otherwise, it looks for the focus in the results.json file in the directory of the source (e.g., 'star0'). 

	Args:
		# fixed arguments: 
		flt
		drz
		obs
		tar
		# user arguments:
		source='qso'
		freeparams=['dx', 'dy', 'scale', 'sigma']
		params_range={} : dictionary of ranges (tuple) that freeparams can take
		neg_penal=1. (float): factor to penalize negative residuals
		refocus=false
		source_focus='star0'
		nx=64 (int): cutout size in x
		ny=64 (int): cutout size in y
		pixsize=0.13 (float): in arcsec

	"""

	# setting
	diameter = 6
	focus = 0.
	subsample = 6

	# run
	if not os.path.isfile(flt.fp_skysub):
		flt.make_flt_skysub()

	# focus
	if refocus:
		s = flt.sources[source_focus]

		if not os.path.isdir(s.directory):
			os.mkdir(s.directory)

		dir_psf = s.directory+'psf/'
		if not os.path.isdir(dir_psf):
			os.mkdir(dir_psf)

		tpsf = tinyfit.tinypsf.tinypsf(camera=obs.camera, filter=obs.filter, position=[s.x, s.y], spectrum_form=s.spectrum_form, spectrum_type=s.spectrum_type, diameter=diameter, focus=focus, subsample=subsample, fn='psf', dir_out=dir_psf, pixsize=pixsize)

		f = tinyfit.imgfitter.imgfitter(filename=flt.fp_skysub, pixsize=pixsize, nx=nx, ny=ny)
		f.set_model(tinypsf=tpsf)
		f.hyperfit(x=s.x, y=s.y, freeparams=['dx', 'dy', 'scale', 'sigma'], freehyperparams=['focus'])
		# write outputs
		f.img_crop.writeto(s.directory+'img_crop.fits', overwrite=True)
		f.img_crop_bestfit.writeto(s.directory+'img_crop_bestfit.fits', overwrite=True)
		f.img_crop_residual.writeto(s.directory+'img_crop_residual.fits', overwrite=True)
		f.result.save(s.directory+'result.json')
		focus_bestfit = f.result.hyperparams.focus
	else: 
		# get focus
		f = tinyfit.imgfitter.imgfitter(filename=flt.fp_skysub, pixsize=pixsize, nx=nx, ny=ny)
		f.result.load(flt.sources[source_focus].directory+'result.json')
		focus_bestfit = f.result.hyperparams.focus

	# source psf fit
	s = flt.sources[source]
	if not os.path.isdir(s.directory):
		os.mkdir(s.directory)

	dir_psf = s.directory+'psf/'
	if not os.path.isdir(dir_psf):
		os.mkdir(dir_psf)

	tpsf = tinyfit.tinypsf.tinypsf(camera=obs.camera, filter=obs.filter, position=[s.x, s.y], spectrum_form=s.spectrum_form, spectrum_type=s.spectrum_type, diameter=diameter, focus=focus_bestfit, subsample=subsample, fn='psf', dir_out=dir_psf, pixsize=pixsize)

	f = tinyfit.imgfitter.imgfitter(filename=flt.fp_skysub, pixsize=pixsize, nx=nx, ny=ny)
	f.set_model(tinypsf=tpsf)
	tpsf.write_params(s.directory+'tiny_params.json')
	f.fit(x=s.x, y=s.y, freeparams=freeparams, params_range=params_range, neg_penal=neg_penal)

	# write outputs
	f.img_crop.writeto(s.directory+'img_crop.fits', overwrite=True)
	f.img_crop_bestfit.writeto(s.directory+'img_crop_bestfit.fits', overwrite=True)
	f.img_crop_residual.writeto(s.directory+'img_crop_residual.fits', overwrite=True)
	f.result.save(s.directory+'result.json')

	hdus_bestfit = f.get_hdus_bestfit()
	hdus_bestfit.writeto(s.directory+'flt_psf.fits', overwrite=True)


def funcdrz_psfsub(drz, obs, tar, source='qso', nx=64, ny=64, pixsize=0.13, final_pixfrac=1., fn_final_refimage='drz.fits'):
	""" astrodrizzle flt_psf_{source}.fits to produce drz_psf_{source}.fits and subtract it from drz.fits to get drz_psfsub_source.fits. Crop drz.fits and drz_psfsub_{source}.fits to get drz_crop{source}.fits and drz_psfsub_{source}_crop{source}.fits

	Args:
		# fixed arguments: 
		drz
		obs
		tar
		# user arguments:
		source='qso'
		nx=64 (int): cutout size in x
		ny=64 (int): cutout size in y
		pixsize=0.13 (float): in arcsec
		final_pixfrac=1. (float): drizzle drop size
	"""
	s = drz.sources[source]
	if not os.path.isdir(s.directory):
		os.mkdir(s.directory)

	drz.fp_psf = s.directory+'drz_psf.fits'
	drz.fp_psfsub = s.directory+'drz_psfsub.fits'
	fp_flt_psfs = [flt.sources[source].directory+'flt_psf.fits' for flt in drz.flts]

	adz.AstroDrizzle(input=fp_flt_psfs, output=drz.fp_psf, static=False, build=True, skysub=False, driz_cr=False, preserve=False, runfile=drz.directory+'astrodrizzle_psfsub.log', final_pixfrac=final_pixfrac, final_refimage=drz.directory+fn_final_refimage)

	# PSF subtraction
	hdus = fits.open(drz.fp_local)
	hdus_psf = fits.open(drz.fp_psf)
	data_psfsub = hdus[1].data - hdus_psf[1].data
	hdus[1].data = data_psfsub
	hdus[1].header['HISTORY'] = 'IMAGE REPLACED BY PSF SUBTRACTED IMAGE'
	hdus[0].header['HISTORY'] = 'SCI IMAGE REPLACED BY PSF SUBTRACTED IMAGE'
	hdus.writeto(drz.fp_psfsub, overwrite=True)

	# crop cutouts
	f = tinyfit.imgfitter.imgfitter(filename=drz.directory+'drz.fits', nx=nx, ny=ny, pixsize=pixsize)
	f._crop(xc=s.x, yc=s.y)
	f.img_crop.writeto(s.directory+'drz_crop.fits')

	f_psub = tinyfit.imgfitter.imgfitter(filename=drz.fp_psfsub, nx=nx, ny=ny, pixsize=pixsize)
	f_psub._crop(xc=s.x, yc=s.y)
	f_psub.img_crop.writeto(s.directory+'drz_psfsub_crop.fits')

	f_psf = tinyfit.imgfitter.imgfitter(filename=drz.fp_psf, nx=nx, ny=ny, pixsize=pixsize)
	f_psf._crop(xc=s.x, yc=s.y)
	f_psf.img_crop.writeto(s.directory+'drz_psf_crop.fits')



def funcdrz_plot(drz, obs, tar, source='qso'):
	""" make ratio plot with 3 panels: 'drz_crop' 'drz_psfsub_crop' 'ratio of the two'

	Args:
		# fixed arguments: 
		drz
		obs
		tar
		# user arguments:
		source='qso'
	"""

	s = drz.sources[source]
	if not os.path.isdir(s.directory):
		os.mkdir(s.directory)

	fn_crop = s.directory+'drz_crop.fits'
	fn_psfsub_crop = s.directory+'drz_psfsub_crop.fits'
	fn_plot = s.directory+'drz_crop_ratio.pdf'

	v = tinyfit.visual.visual(fn=fn_crop)

	v.ratioplot(fn=fn_psfsub_crop, stretch='linear', cmap=plt.cm.jet, fn_out=fn_plot, colorbar=True)
	plt.close('all')
