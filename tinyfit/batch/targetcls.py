# -*- coding: utf-8 -*-

"""
contain class for targets, observations, drz, flt, and sources. 
"""

import os
import sys
import copy
import shutil
import numpy as np
from astropy import wcs
from astropy.io import fits
import astropy.table as at
import gzip
import jinja2 as jin
import numbers


def make_roadmap(fp_tab_target, fp_tab_observations, fp_tab_sources, source_names, observation_names=None, fp='roadmap.json', key='obj_name', dir_data='./', dir_local='./'):
	""" make roadmap json file from input tables

	Note: 
		each table has to have the same number of rows. Each table has to have a key column with name, e.g., 'obj_name', as set by args 'key', for unique identification of targets. 

		All drz and flt fitz file should locate in one single directory dir_data. 

	Args:
		fp_tab_target (str): 
			path to input target table. Columns: 'ra', 'dec'.
		fp_tab_observations (list of str): 
			list of paths to input observation tables. Columns: 'camera', 'filter', 'fn_drzs'.
		fp_tab_sources (list of str): 
			list of paths to input source tables. Columns: 'ra', 'dec', 'spectral_form', 'spectral_type'
		source_names (list of str): list of source names, same length as fp_tab_sources. 
		observation_names=None (list of str, optional): 
			list of observation names, same length as fp_tab_observations. 
			Default: 'obs0', 'obs1', etc.
		fp='roadmap.json' (str, optional): path to output json file. 
		key='obj_name' (str): column name of the key column. 
		dir_data='./' (str): directory where fits files are taken from. 
		dir_local='./' (str): directory where new local data tree will be built. 

	Return:
		(bool): True if succes
	"""
	# setting
	tab_target = at.Table.read(fp_tab_target)
	tab_observations = [at.Table.read(f) for f in fp_tab_observations]
	tab_sources = [at.Table.read(f) for f in fp_tab_sources]

	if observation_names is None:
		observation_names = ['obs'+str(i) for i in range(len(fp_tab_observations))]

	# make sure each table has corresponding rows.
	tab_target.sort(key)
	lst_target = list(tab_target[key])
	for tab_observation in tab_observations:
		tab_observation.sort(key)
		assert list(tab_observation[key]) == lst_target

	for tab_observation in tab_observations:
		tab_observation.sort(key)
		assert list(tab_observation[key]) == lst_target

	for tab_source in tab_sources:
		tab_source.sort(key)
		assert list(tab_source[key]) == lst_target

	# run
	aux_cols_target = tab_target.colnames
	[aux_cols_target.remove(col) for col in [key, 'ra', 'dec']]

	targets = []
	for i, row_target in enumerate(tab_target):
		aux = {col: row_target[col] for col in aux_cols_target}
		target = Target(name=row_target[key], ra=row_target['ra'], dec=row_target['dec'], dir_parent=dir_local, **aux)
		sources = {source_names[j]: Source(name=source_names[j], ra=tab_source[i]['ra'], 
					dec=tab_source[i]['dec'], 
					spectrum_form=tab_source[i]['spectrum_form'], 
					spectrum_type=tab_source[i]['spectrum_type']
					) for j, tab_source in enumerate(tab_sources)}

		observations = []
		for k, tab_observation in enumerate(tab_observations):
			observation = Observation(name=observation_names[k], 
									camera=tab_observation[k]['camera'], 
									filter=tab_observation[k]['filter'], 
									dir_parent=target.directory)

			fn_drzs = tab_observation[i]['fn_drzs'].split(',')
			drz_names = ['drz'+str(l) for l in range(len(fn_drzs))]
			drzs = [DRZ(name=drz_names[l], fp=dir_data+fn_drzs[l], sources=sources, dir_parent=observation.directory) for l in range(len(fn_drzs))]
			observation.drzs = drzs
			observations = observations+[observation]
		target.observations = observations
		targets = targets +[target]

	render_roadmap(fp, targets)
	return os.path.isfile(fp)


def render_roadmap(fp, targets):
	"""
	write targets as json file to fp
	"""
	fp_template = 'roadmap_template.json'
	env = jin.Environment(loader=jin.FileSystemLoader(getlocalpath()))
	env.filters['quote'] = quote_jfilter
	temp = env.get_template(fp_template)
	targets_json = temp.render(targets=targets)

	with open(fp, 'w') as file:
		file.write(targets_json)

def getlocalpath():
	"""
	return path to filter direcotry
	"""
	path = os.path.dirname(sys.modules[__name__].__file__)
	if path == '': path ='.'
	return path+'/'


def quote_jfilter(string):
	if isinstance(string, numbers.Number):
		return string
	else:
		try: 
			return "\""+string+"\""
		except:
			return "null"


class Target(object):
	"""Target that has all info for processing psf-subtraction of an target

	Attributes:
		name (str): name of the source. 
		directory (str): path to the working directory
		ra (float): ra
		dec (float): dec
		reference (str): sample source of the target, could be None.
		sources (list of :obj: Source): 
			list of sources associated with the target with attributes ra, dec, spectrum_form, spectrum_type. 
		observations (list of :obj: Observation): 
			list of observations of the target. 
		roadmap (json, optional): json representation of the object. 
	"""
	def __init__(self, dir_parent='./', **kwargs):
		""" initializing 

		Note:
			attribute `directory` is set to dir_parent+name+'/'

		Args: 
			dir_parent='./' (str): path to the parent directory
			either: 
				roadmap (json): json representation of the object containing the following. 
			or: 
				name (str): name of the source. 
				ra (float): ra
				dec (float): dec
				reference (str): sample source
				sources (list of :obj: Source): 
					list of sources associated with the target with attributes ra, dec, spectrum_form, spectrum_type. 
				observations (list of :obj: Observation): 
					list of observations of the target. 
		"""
		if 'roadmap' in kwargs:
			roadmap = kwargs.pop('roadmap', None)
			self.roadmap = copy.copy(roadmap)
			self.name = roadmap.pop('name', None)
			self.directory = dir_parent+self.name+'/'
			self.ra = roadmap.pop('ra', None)
			self.dec = roadmap.pop('dec', None)
			self.reference = roadmap.pop('reference', None)
			self.sources = get_dict_Sources_from_roadmap(roadmap=roadmap.pop('sources', []))
			self.observations = [Observation(dir_parent=self.directory, roadmap=r) for r in roadmap.pop('observations', [])]
		else: 
			self.name = kwargs.pop('name', None)
			self.directory = dir_parent+self.name+'/'
			self.ra = kwargs.pop('ra', None)
			self.dec = kwargs.pop('dec', None)
			self.reference = kwargs.pop('reference', None)
			self.sources = kwargs.pop('sources', {})
			self.observations = kwargs.pop('observations', [])



class Observation(object):
	""" an observation that has name, camera, filter, drzs 

	Attributes:
		name (str): name of the observation. 
		directory (str): path to the working directory
		camera (str): the camera. 
		filter (str): the filter. 
		drzs (list of :obj: DRZ): the drz files associated with the observation. 
		roadmap (json, optional): json representation of the object. 
	"""
	def __init__(self, dir_parent='./', **kwargs):
		""" initializing

		Note:
			attribute `directory` is set to dir_parent+name+'/'

		Args:
			dir_parent='./' (str): path to the parent directory
			either: 
				roadmap (json): json representation of the object containing the following. 
			or: 
				name (str): name of the observation. 
				camera (str): Following tinytim convention should be one of the following: 
					'wfpc1', 'wfpc1_planetary', 'wfpc1_foc_f48', 'wfpc1_foc_f48', 'wfpc2', 'wfpc2_planetary', 'wfpc2_foc_f48', 'wfpc2_foc_f48', 'nicmos1_precryo', 'nicmos2_precryo', 'nicmos3_precryo', 'stis_ccd', 'stis_nuv', 'stis_fuv', 'acs_widefield', 'acs_highres', 'acs_coronoffspot', 'acs_solarblind', 'nicmos1_cryo', 'nicmos2_cryo', 'nicmos3_cryo', 'wfc3_uvis', 'wfc3_ir',

				filter (str): the filter, e.g., 'f160w', see tinytim.  
				drzs (list of :obj: DRZ): the drz files associated with the observation. 
		"""
		if 'roadmap' in kwargs:
			roadmap = kwargs.pop('roadmap', None)
			self.roadmap = copy.copy(roadmap)
			self.name = roadmap.pop('name', None)
			self.directory = dir_parent+self.name+'/'
			self.camera = roadmap.pop('camera', None)
			self.filter = roadmap.pop('filter', None)
			self.drzs = [DRZ(dir_parent=self.directory, roadmap=r) for r in roadmap.pop('drzs', [])]
		else: 
			self.name = kwargs.pop('name', None)
			self.directory = dir_parent+self.name+'/'
			self.camera = kwargs.pop('camera', None)
			self.filter = kwargs.pop('filter', None)
			self.drzs = kwargs.pop('drzs', [])



class DRZ(object):
	"""a hst drz file that has name, fp, and a list of flt files. 

	Attributes:
		name (str): name of the drz file, can be arbitrary. 
		directory (str): path to the working directory
		fp (str): file name (including path) to the drz file. 
		fp_local (str): file path in local data tree, set to direcotry+'drz.fits'.
		wcs (:obj: 'astropy.wcs.WCS')
		flts (:obj: 'list' of :obj: Flt)
		sources (list of :obj: FITSource, optional)
		roadmap (json, optional): json representation of the object. 

	Methods:
		copyfile()
	"""
	def __init__(self, dir_parent='./', **kwargs):
		""" Initialize drz object. 

		Note: 
			attribute `directory` is set to dir_parent+name+'/'

			wcs is read from the header. 

			The FLT files will be automatically identified from drz header if not specified. For automated identification, the flt files should locate in the same directory as the drz file. The flts are named as flt0, flt1, ..., etc. 

			If fp does not exist, it will attemp to unzip a file named fp+'.gz'. 

		Args: 
			dir_parent='./' (str): path to the parent directory
			either: 
				roadmap (json): 
					json representation of the object containing the following. If it does not contain flts info, it will be automatically extracted from the header of the drz file. 
			or: 
				name (str): name of the drz file, can be arbitrary. 
				fp (str): file name (including path) to the drz file. 
				sources = {} (list of :obj: Source or FITSource, optional)
				flts = [] (:obj: 'list' of :obj: Flt, optional): 
					If not specified, it will be automatically extracted from the header of the drz file. 
		"""
		if 'roadmap' in kwargs:
			roadmap = kwargs.pop('roadmap', None)
			self.roadmap = copy.copy(roadmap)
			self.name = roadmap.pop('name', None)
			self.directory = dir_parent+self.name+'/'
			self.fp = roadmap.pop('fp', None)
			# unzip fp
			if not os.path.isfile(self.fp):
				unzip_gz(self.fp+'.gz', self.fp)
			self.wcs = wcs.WCS(fits.getheader(self.fp, ext=1))
			self.flts = [FLT(dir_parent=self.directory, roadmap=r) for r in roadmap.pop('flts', [])]
			self.sources = get_dict_FITSources_from_roadmap(roadmap=roadmap.pop('sources', {}), wcs=self.wcs)

		else: 
			self.name = kwargs.pop('name', None)
			self.directory = dir_parent+self.name+'/'
			self.fp = kwargs.pop('fp', None)
			# unzip fp
			if not os.path.isfile(self.fp):
				unzip_gz(self.fp+'.gz', self.fp)
			self.wcs = wcs.WCS(fits.getheader(self.fp, ext=1))
			self.flts = kwargs.pop('flts', [])
			sources = kwargs.pop('sources', {})
			self.set_fitsources(sources)
			# self.sources = {name: FITSource(wcs=self.wcs, name=name, ra=source.ra, dec=source.dec, ) for name, source in sources.items()}

		# automated identification of the flt files. 
		if self.flts == []:
			self._initialize_flts_from_header()

		self.fp_local = self.directory+'drz.fits'


	def set_fitsources(self, sources):
		""" setting attribute sources for drz and all the flts from argument and enforcing that x, y are calculated from the corresponding wcs. Auxiliary attributes of sources are also propagated over. 
		"""
		# setting list of additional auxiliary attributes from an arbitrary source
		aux_attr = list(next(iter(sources.values())).__dict__.keys())
		[aux_attr.remove(key) for key in ['name', 'ra', 'dec']]

		self.sources = {name: FITSource(name=name, ra=source.ra, dec=source.dec, wcs=self.wcs, **{a: getattr(source, a) for a in aux_attr}) for name, source in sources.items()}

		for flt in self.flts:
			flt.set_fitsources(sources)


	def copyfile(self):
		""" copy fits file from fp to local location fp_local. 
		"""
		if self.fp_local != self.fp:
			if not os.path.isdir(self.directory):
				os.makedirs(self.directory)
			shutil.copy(self.fp, self.fp_local)


	def _initialize_flts_from_header(self, dir_flt=None):
		""" inititalize flts from the header of the drz file

		Args: 
			dir_flt (str, optional): 
				path of the flt parent directory. Default is set to the parent directory of the drz file. 
		"""
		header = fits.getheader(self.fp)
		nflt = header['NDRIZIM']
		flt_fns = [header['D00{}DATA'.format(str(i+1))].split('[')[0] for i in range(nflt)]

		if dir_flt is None:
			dir_flt = os.path.dirname(self.fp)+'/'
		
		self.flts = [FLT(name='flt{}'.format(str(i)), fp=dir_flt+flt_fns[i], sources=self.sources, dir_parent=self.directory) for i in range(nflt)]


class FLT(object):
	"""an hst flt file that has name, fp (filepath), and optionally a list of sources. 

	Attributes:
		name (str): name of the flt file, can be arbitrary. 
		directory (str): path to the working directory
		fp (str): file name (including path) to the flt file. 
		fp_local (str): file path in local data tree, set to direcotry+'flt.fits'.
		wcs (:obj: 'astropy.wcs.WCS')
		sources (list of :obj: FITSource, optional)
		roadmap (json, optional): json representation of the object. 

	Methods:
		copyfile()
	"""
	def __init__(self, dir_parent='./', **kwargs):
		""" Initialize flt object. 

		Note: 
			attribute `directory` is set to dir_parent+name+'/'
			The wcs is read from the header. 
			If fp does not exist, it will attemp to unzip a file named fp+'.gz'. 

		Args: 
			dir_parent='./' (str): path to the parent directory
			either: 
				roadmap (json): json representation of the object containing the following. 
			or: 
				name (str): name of the flt file, can be arbitrary. 
				fp (str): file name (including path) to the flt file. 
				sources = {} (list of :obj: Source or FITSource, optional)
		"""
		if 'roadmap' in kwargs:
			roadmap = kwargs.pop('roadmap', None)
			self.roadmap = copy.copy(roadmap)
			self.name = roadmap.pop('name', None)
			self.directory = dir_parent+self.name+'/'
			self.fp = roadmap.pop('fp', None)
			# unzip fp
			if not os.path.isfile(self.fp):
				unzip_gz(self.fp+'.gz', self.fp)
			self.wcs = wcs.WCS(fits.getheader(self.fp, ext=1))
			self.sources = get_dict_FITSources_from_roadmap(roadmap=roadmap.pop('sources', {}), wcs=self.wcs)
		else: 
			self.name = kwargs.pop('name', None)
			self.directory = dir_parent+self.name+'/'
			self.fp = kwargs.pop('fp', None)
			# unzip fp
			if not os.path.isfile(self.fp):
				unzip_gz(self.fp+'.gz', self.fp)
			self.wcs = wcs.WCS(fits.getheader(self.fp, ext=1), fits.open(self.fp))
			sources = kwargs.pop('sources', {})
			self.set_fitsources(sources)
			# self.sources = {name: FITSource(name=name, ra=source.ra, dec=source.dec, wcs=self.wcs, ) for name, source in sources.items()}

		self.fp_local = self.directory+'flt.fits'


	def set_fitsources(self, sources):
		""" setting attribute sources from argument and enforcing that x, y are calculated from self.wcs. Auxiliary attributes of sources are also propagated over. 
		"""
		# setting list of additional auxiliary attributes from an arbitrary source
		aux_attr = list(next(iter(sources.values())).__dict__.keys())
		[aux_attr.remove(key) for key in ['name', 'ra', 'dec']]

		self.sources = {name: FITSource(name=name, ra=source.ra, dec=source.dec, wcs=self.wcs, **{a: getattr(source, a) for a in aux_attr}) for name, source in sources.items()}


	def copyfile(self):
		""" copy fits file from fp to local location fp_local. 
		"""
		if self.fp_local != self.fp:
			if not os.path.isdir(self.directory):
				os.makedirs(self.directory)
			shutil.copy(self.fp, self.fp_local)



class Source(object):
	""" a source that has name, ra, dec, and optionally other attributes. 

	Attributes:
		name (str): name of the source. 
		# directory (str): path to the working directory
		ra (float): ra
		dec (float): dec
	"""
	def __init__(self, **kwargs):
		""" initializing 
		Args: 
			name (str): name of the source. 
			ra (float): ra
			dec (float): dec
			# dir_parent='./' (str): path to the parent directory
			**kwargs: 'key' 'value' pairs that will be set as attributes. 
		"""
		self.name = kwargs.pop('name', None)
		# self.directory = kwargs.pop('dir_parent', './')+self.name+'/'
		self.ra = kwargs.pop('ra', None)
		self.dec = kwargs.pop('dec', None)		
		for key, value in kwargs.items():
			setattr(self, key, value)


class FITSource(Source):
	"""a source that has name, ra, dec and corresponding x, y position in the fits file

	Note: 
		If x, y is not provided as kwargs, x, y is automatically inferred from wcs and ra, dec. 

	Attributes:
		name (str): name of the source. 
		# directory (str): path to the working directory
		ra (float): ra
		dec (float): dec
		x (int): x position in fits file
		y (int): y position in fits file
	"""
	def __init__(self, wcs, **kwargs):
		""" initializing
		Args: 
			wcs (:obj: 'astropy.wcs.WCS')
			# dir_parent='./' (str): path to the parent directory
			name (str): name of the source. 
			ra (float): ra
			dec (float): dec
			x (int): x position in fits file
			y (int): y position in fits file
		"""

		super(self.__class__, self).__init__(**kwargs)

		# set origin to 0 to comply with numpy convention
		if (not hasattr(self, 'x')) | (not hasattr(self, 'y')):
			x, y = np.round(wcs.wcs_world2pix(np.array([[self.ra, self.dec]]), 0)[0])
			self.x, self.y = int(x), int(y)



def get_dict_FITSources_from_roadmap(roadmap, wcs):
	"""
	translate roadmap of sources to dictionary of Source objects. 

	Args: 
		roadmap (dict): For example:	
			{ 
				"qso":{
					"ra": 128.000833,
					"dec": 16.250111,
					"spectrum_form": "powerlaw_nu",
					"spectrum_type": -1.33441542369
				}, 
				"star0":{
					"ra": 128.01769053,
					"dec": 16.24563831,
					"spectrum_form": "stellar",
					"spectrum_type": "g8v"
				}, 
			}
		wcs (:obj: 'astropy.wcs.WCS')

	Return: 
		{dict of :obj: 'FITSource'}
	"""
	return {name: FITSource(wcs=wcs, name=name, **value) for name, value in roadmap.items()}


def get_dict_Sources_from_roadmap(roadmap):
	"""
	translate roadmap of sources to dictionary of Source objects. 

	Args: 
		roadmap (dict): For example:	
			{ 
				"qso":{
					"ra": 128.000833,
					"dec": 16.250111,
					"spectrum_form": "powerlaw_nu",
					"spectrum_type": -1.33441542369
				}, 
				"star0":{
					"ra": 128.01769053,
					"dec": 16.24563831,
					"spectrum_form": "stellar",
					"spectrum_type": "g8v"
				}, 
			}

	Return: 
		{dict of :obj: 'Source'}
	"""
	return {name: Source(name=name, **value) for name, value in roadmap.items()}


def unzip_gz(fp_in, fp_out):
	""" unzip gz file fp_in to fp_out """
	print('unzipping', fp_in)
	with gzip.open(fp_in, 'rb') as f_in:
		with open(fp_out, 'wb') as f_out:
			shutil.copyfileobj(f_in, f_out)
