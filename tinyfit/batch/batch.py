# -*- coding: utf-8 -*-

"""
batch runs psf fitting on targets
"""
import os
import json
import pandas as pd
import copy

from . import targetcls
from .targetcls import make_roadmap
from .targetcls import render_roadmap

class Batch(object):
	""" a batch of targets for running psf subtraction 

	Example: 
		>>> from tinyfit.batch import Batch
		>>> b = Batch('roadmap.json', directory='./')

	Attributes:
		roadmap (list of dict): loaded roadmap json file. 
		directory (str): path to the working directory
		targets (list of :obj: Target): targets to run operations on

	Methods:
		build()
		write_roadmap()
	"""
	def __init__(self, fp_roadmap, directory='./'):
		""" initializing

		Args: 
			fp_roadmap (str): path to json file that contains targets information
			directory='./' (str): path of the directory for operations to run to
		"""
		with open(fp_roadmap) as f:
			self.roadmap = json.load(f)

		self.directory = directory
		
		self.targets = [targetcls.Target(roadmap=r, dir_parent=self.directory) for r in self.roadmap]


	def build(self, copydrz=True, copyflt=True):
		""" create directory tree that contains directories for each target, observation, drz, flt and source of flt. copy drz and flt fits files to corresponding directories. 

		Args:
			copydrz=True (bool): if false, then not copy drz to local
			copyflt=True (bool): if false, then not copy flt to local

		"""
		for tar in self.targets:
			if not os.path.isdir(tar.directory):
				os.mkdir(tar.directory)
			for obs in tar.observations:
				if not os.path.isdir(obs.directory):
					os.mkdir(obs.directory)
				for drz in obs.drzs:
					if not os.path.isdir(drz.directory):
						os.mkdir(drz.directory)
					if copydrz:
						drz.copyfile()
					for flt in drz.flts:
						if not os.path.isdir(flt.directory):
							os.mkdir(flt.directory)
						if copyflt:
							flt.copyfile()


	def itertar(self, func, **kwargs):
		""" call func with arguments tar iteratively for each of the tar 

		Args:
			func (function): with arguments tar. 
			**kwargs : additional arguments to pass to func
		"""
		for tar in self.targets:
			func(tar=tar, **kwargs)


	def iterdrz(self, func, **kwargs):
		""" call func with arguments drz, obs, tar iteratively for each of the drz 

		Args:
			func (function): with arguments drz, obs, tar. 
			**kwargs : additional arguments to pass to func
		"""
		for tar in self.targets:
			for obs in tar.observations:
				for drz in obs.drzs:
					func(drz=drz, obs=obs, tar=tar, **kwargs)


	def iterflt(self, func, **kwargs):
		""" call func with arguments flt, drz, obs, tar iteratively for each of the flt 

		Args:
			func (function): with arguments flt, drz, obs, tar. 
			**kwargs : additional arguments to pass to func
		"""
		for tar in self.targets:
			for obs in tar.observations:
				for drz in obs.drzs:
					for flt in drz.flts:
						func(flt=flt, drz=drz, obs=obs, tar=tar, **kwargs)


	def compiledrz(self, fn, fp_out=None, observation_names=[]):
		"""
		return compiled csv file from the corresponding file in each of the drz directory. 

		Note: 
			new columns tar, observation, drz will be added in front. 

		Args:
			fn (str): file name of the csv file, for example, 'number.csv'. 
			fp_out=None (str): if set to str then write result as csv file to fp_out. 
			observations_names=[] (list): if set to list of strings, e.g., ['obs0', ] then only those observations will be ran. 

		Return:
			(:obj: pandas.DataFrame)
		"""
		df_compiled = pd.DataFrame()
		for tar in self.targets:
			for obs in tar.observations:
				if (len(observation_names)<0) | (obs.name in observation_names): 
					for drz in obs.drzs:
						df = pd.read_csv(drz.directory+fn)
						df.insert(loc=0, column='target', value=tar.name)
						df.insert(loc=1, column='observation', value=obs.name)
						df.insert(loc=2, column='drz', value=drz.name)
						df_compiled = pd.concat([df_compiled, df], ignore_index=True)

		if fp_out is not None:
			df_compiled.to_csv(fp_out, index=False)

		return df_compiled


	def compiletar(self, fn, fp_out=None):
		"""
		return compiled csv file from the corresponding file in each of the tar directory. 

		Note: 
			new column tar will be added in front. 

		Args:
			fn (str): file name of the csv file, for example, 'number.csv'. 
			fp_out=None (str): if set to str then write result as csv file to fp_out. 

		Return:
			(:obj: pandas.DataFrame)
		"""
		df_compiled = pd.DataFrame()
		for tar in self.targets:
			print(tar.name)
			df = pd.read_csv(tar.directory+fn)
			# df_compiled = pd.concat([df_compiled, df], ignore_index=True)
			df.insert(loc=0, column='target', value=tar.name)
			df_compiled = df_compiled.append(df)

			# pd.concat([df_compiled, df], ignore_index=True)


		if fp_out is not None:
			df_compiled.to_csv(fp_out, index=False)

		return df_compiled

	def compiledrz_source(self, fn, source='qso', fp_out=None):
		"""
		return compiled csv file from the corresponding source directory in each of the drz. 

		Note: 
			new columns tar, observation, drz, source will be added in front. 

		Args:
			fn (str): file name of the csv file, for example, 'number.csv'. 
			source='qso' (str): the source under drz to be compiled, for example, 'qso'. 
			fp_out=None (str): if set to str then write result as csv file to fp_out. 

		Return:
			(:obj: pandas.DataFrame)
		"""
		df_compiled = pd.DataFrame()
		for tar in self.targets:
			for obs in tar.observations:
				for drz in obs.drzs:
					s = drz.sources[source]
					df = pd.read_csv(s.directory+fn)
					df.insert(loc=0, column='target', value=tar.name)
					df.insert(loc=1, column='observation', value=obs.name)
					df.insert(loc=2, column='drz', value=drz.name)
					df.insert(loc=3, column='source', value=source)
					df_compiled = pd.concat([df_compiled, df], ignore_index=True)

		if fp_out is not None:
			df_compiled.to_csv(fp_out, index=False)

		return df_compiled


	def write_roadmap(self, fp=None):
		""" write roadmap representing the Batch as json to file. 

		Args:
			fp=None (str): file name to write to, default: self.directory+'roadmap.json'
		"""
		if fp is None:
			fp = self.directory+'roadmap.json'
		
		render_roadmap(fp, targets=self.targets)


	def merge_drzs(self):
		"""
		for each obs in the batch merge all the drzs into one single master drz that contains all the flts. 

		Notes: 
			The new drz will have no remote fp to copy file from. To build the batch, please use b.build(copydrz=False)
		"""
		for tar in self.targets:
			for obs in tar.observations:
				flts_all = []
				for drz in obs.drzs:
					flts_all += drz.flts

				drz_master = copy.deepcopy(obs.drzs[0])
				sources = copy.deepcopy(obs.drzs[0].sources)
				drz_master.name = 'drz_master'
				drz_master.directory = obs.directory+drz_master.name+'/'
				drz_master.fp = ''
				drz_master.fp_local = drz_master.directory+'drz.fits'

				drz_master.flts = [targetcls.FLT(name='flt{}'.format(str(i)), fp=flts_all[i].fp, sources={}, dir_parent=drz_master.directory) for i in range(len(flts_all))] # source will be updated later

				obs.drzs = [drz_master]
				obs.drzs[0].set_fitsources(sources)
				print(obs.drzs[0].directory)