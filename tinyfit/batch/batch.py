# -*- coding: utf-8 -*-

"""
batch runs psf fitting on targets
"""
import os
import json

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


	def build(self):
		""" create directory tree that contains directories for each target, observation, drz, flt and source of flt. copy drz and flt fits files to corresponding directories. 

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
					drz.copyfile()
					for flt in drz.flts:
						if not os.path.isdir(flt.directory):
							os.mkdir(flt.directory)
						flt.copyfile()


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

	def write_roadmap(self, fp=None):
		""" write roadmap representing the Batch as json to file. 

		Args:
			fp=None (str): file name to write to, default: self.directory+'roadmap.json'
		"""
		if fp is None:
			fp = self.directory+'roadmap.json'
		
		render_roadmap(fp, targets=self.targets)

