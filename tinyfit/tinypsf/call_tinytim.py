"""
collection of functions for calling tinytim

	tiny1()
	tiny2()
	tiny3()
"""

import os
import pexpect
import time

from . import tiny_dictionary as td


def tiny1(dir_code, fn, camera='wfc3_ir', detector=0, position=[500, 500], filter='f160w', spectrum_form='stellar', spectrum_type='f8v', diameter=3, focus=0.0, rootname='temporary', ):
	"""
	call tiny1 with input params to write params file

	Example
	-------

	tiny1(dir_code='where/is/tiny/tinytim/code/', fn='test.param', camera='wfc3_ir, position=[500, 500], filter='f160w, spectrum_form='stellar', spectrum_type='f8v', diameter=3, focus=0.0, rootname='temporary')

	Return
	------
	status (bool)

	Output
	------
	ascii file named fn, that contains params required for tiny2. 

	Params
	------
	dir_code (str):
		location of the tinytim code where executables tiny1, tiny2, and tiny3 reside. 
		e.g., '~/Documents/tinytim/'

	fn (str):
		file path to the param file to be produced,
		e.g., 'test.param'

	camera='wfc3_ir' (str):
		Choose from --
		'wfpc1', 'wfpc1_planetary', 'wfpc1_foc_f48', 'wfpc1_foc_f48', 'wfpc2', 'wfpc2_planetary', 'wfpc2_foc_f48', 'wfpc2_foc_f48', 'nicmos1_precryo', 'nicmos2_precryo', 'nicmos3_precryo', 'stis_ccd', 'stis_nuv', 'stis_fuv', 'acs_widefield', 'acs_highres', 'acs_coronoffspot', 'acs_solarblind', 'nicmos1_cryo', 'nicmos2_cryo', 'nicmos3_cryo', 'wfc3_uvis', 'wfc3_ir', 

	detector=0 int:
		which detector (1 or 2), only used when camera == 'acs_widefield'. 

	position=[500, 500] list of int:
		[x, y]

	filter='f160w' (str):

	spectrum_form='stellar' (str):
		Choose from --
		'stellar', 'blackbody', 'powerlaw_nu', 'powerlaw_lam', 'user'

	spectrum_type='f8w' (str):
		if spectrum_form == 'stellar': str
			choose: ['o5', 'o8f', 'o6', 'b1v', 'b3v', 'b6v', 'a0v', 'a5v', 'f6v', 'f8v', 'g2v', 'g5v', 'g8v', 'k4v', 'k7v', 'm1.5v', 'm3v', ]

		if spectrum_form == 'blackbody': float
			Enter temperature (Kelvin)

		if spectrum_form == 'powerlaw_nu': float
			F(nu) = nu^alpha 
			Enter the spectral index (alpha)

		if spectrum_form == 'powerlaw_lam': float
			F(lambda) = lambda^beta 
			Enter the spectral index (beta)

		if spectral_form == 'user': str
			NOTE : Spectrum MUST have < 15000 points and be
			in Angstrom and flux pairs (see manual for flux types).
			Enter name of spectrum file

			You must provide an ASCII text file which contains the wavelength (in Angstroms) and corresponding flux, one pair per line, up to 15000 pairs. The first line of the file MUST be one of the following, which tells Tiny Tim what units the fluxes are in : FLAM, FNU, PHOTLAM, or JY, where these are defined as :

				FLAM 	ergs Angstrom-1
				FNU 	ergs Hz-1
				PHOTLAM photons Angstrom-1
				JY 		Janskys (W Hz-1)

			For example: 
				PHOTLAM
				100.0 1.0e-3
				120.0 1.3e-3

	diameter=3 (float):
		What diameter should your PSF be (in arcseconds)? 

	focus = 0.0 (float):
		Focus, secondary mirror despace? [microns]

	rootname = 'temporary' (str):
		Rootname of PSF image files (no extension). It's recommended to use the absolute path to where one wants the the PSF images to be. 

		The output of tiny2 will be, e.g., 
			temporary.tt3
			temporary00_psf.fits

		The output of tiny3 will be, e.g., 
			temporary300.fits
	"""

	# e.g., 
	# command = '/Users/aisun/Documents/astro/algorithm/tinytim/tinytim-7.5/tiny1 j1652.param'
	command = dir_code+'tiny1 '+fn

	# Set sleeping time in between prompts. Otherwise tinytim may not modify file. 
	n = 0.01

	p = pexpect.spawn(command)
	time.sleep(n)

	# camera
	p.expect('Choice : ')
	p.sendline(str(td.dict_camera[camera]))
	time.sleep(n)

	# detector
	if camera == 'acs_widefield':
		p.expect(' : ')
		p.sendline(str(detector))
		time.sleep(n)

	# position
	p.expect('Position : ')
	p.sendline('{} {}'.format(str(position[0]), str(position[1])))
	time.sleep(n)
	# filter
	p.expect('Filter : ')
	p.sendline(filter)
	time.sleep(n)
	# spectrum 
	p.expect('Choice : ')
	p.sendline(str(td.dict_spectrum_form[spectrum_form]))
	time.sleep(n)
	# spectrum type 
	if spectrum_form=='stellar':
		p.expect('Enter spectrum # : ')
		p.sendline(str(td.dict_spectrum_stellar[spectrum_type]))
	elif spectrum_form=='blackbody':
		p.expect('Enter temperature (Kelvin) : ')
		p.sendline(str(spectrum_type))
	elif spectrum_form=='powerlaw_nu':
		p.expect('Enter the spectral index (alpha) : ')
		p.sendline(str(spectrum_type))
	elif spectrum_form=='powerlaw_lam':
		p.expect('Enter the spectral index (beta) : ')
		p.sendline(str(spectrum_type))
	elif spectrum_form=='user':
		p.expect('Enter name of spectrum file : ')
		p.sendline(str(spectrum_type))
	else:
		raise Exception("Spectrum type not understood. ")
	time.sleep(n)

	# diameter
	p.expect('(in arcseconds)? : ')
	p.sendline(str(diameter))
	time.sleep(n)

	# focus
	p.expect(': ')
	p.sendline(str(focus))
	time.sleep(n)

	# rootname
	p.expect(' : ')
	p.sendline(rootname)
	time.sleep(n)

	status = os.path.isfile(fn)
	return status


def tiny2(dir_code, fn, rootname='temporary'):
	"""
	call tiny2 with input params to write intermediate files for tiny3

	Example
	-------
	tiny2(dir_code='where/is/tiny/tinytim/code/', fn='test.param')

	Return
	------
	status (bool)

	Output
	------
	files:
		rootname_tiny1+'.tt3'
		rootname_tiny1+'00_psf.fits'
		where rootname_tiny1 is the rootname used in tiny1. 

	Params
	------
	dir_code (str):
		location of the tinytim code where executables tiny1, tiny2, and tiny3 reside. 
		e.g., '~/Documents/tinytim/'

	fn (str):
		file path to the param file produced by tiny1, e.g., 'test.param'. 

	"""
	command = dir_code+'tiny2 '+fn
	os.system(command)

	# return status
	status = os.path.isfile(rootname+'.tt3') & os.path.isfile(rootname+'00_psf.fits')
	return status


def tiny3(dir_code, fn, rootname='temporary', subsample=1):
	"""
	call tiny3 with input params to write output psf. 

	Example
	-------
	tiny3(dir_code='where/is/tiny/tinytim/code/', fn='test.param', sumsample=1)

	Return
	------
	status (bool)

	Output
	------
	files:
		rootname_tiny1+'00.fits'
		where rootname_tiny1 is the rootname used in tiny1. 

	Params
	------
	dir_code (str):
		location of the tinytim code where executables tiny1, tiny2, and tiny3 reside. 
		e.g., '~/Documents/tinytim/'

	fn (str):
		file path to the param file produced by tiny1, e.g., 'test.param'. 

	rootname=temporary (str):
		rootname that was entered in tiny1. It's only used to check if the proper files are created. 

	subsample (int):
		subsampling parameter. Set to larger than 1 to increase the sampling. 

	"""

	command = '{dir_code}tiny3 {fn} sub={sub}'.format(dir_code=dir_code, fn=fn, sub=subsample)
	# command = dir_code+'tiny3 '+fn+' sub='+str(subsample)
	os.system(command)

	status = os.path.isfile(rootname+'00.fits')
	return status


