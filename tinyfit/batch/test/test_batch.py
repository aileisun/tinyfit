import pytest
import os
import shutil
import json
import numpy as np
import random
import pandas as pd


from .. import batch
from ..batch import Batch

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

def test_batch_init():
	b = Batch(dir_data+'roadmap_lite.json', directory=dir_testing)
	assert b.directory == dir_testing
	assert len(b.roadmap) == 2
	assert len(b.targets) == 2

	assert b.targets[0].name == "SDSSJ0832+1615"
	assert b.targets[0].reference == "zakamska"
	assert b.targets[0].directory == b.directory+"SDSSJ0832+1615/"
	assert b.targets[0].observations[0].directory == b.directory+"SDSSJ0832+1615/obs0/"
	assert b.targets[0].observations[0].drzs[0].directory == b.directory+"SDSSJ0832+1615/obs0/drz0/"
	assert b.targets[0].observations[0].drzs[0].flts[0].directory == b.directory+"SDSSJ0832+1615/obs0/drz0/flt0/"

	assert b.targets[0].observations[0].drzs[0].flts[0].sources['qso'].name == 'qso'
	assert b.targets[0].observations[0].drzs[0].flts[0].sources['qso'].x == 556
	stype = -1.3344154236922656
	assert np.absolute(b.targets[0].observations[0].drzs[0].flts[0].sources['qso'].spectrum_type - stype)/stype < 1.e-3
	assert len(b.targets[0].sources) == len(b.targets[0].observations[0].drzs[0].flts[0].sources)


def test_batch_write_roadmap():
	"""
	test that write_roadmap() writes roadmap corresponding to the batch object. 
	"""
	b = Batch(dir_data+'roadmap_lite.json', directory=dir_testing)
	b.write_roadmap()

	assert os.path.isfile(b.directory+'roadmap.json')

	with open(dir_data+'roadmap_lite.json', 'r') as f:
		r0 = json.load(f)

	with open(b.directory+'roadmap.json', 'r') as f:
		r1 = json.load(f)

	assert r0 == r1


def test_batch_build():
	"""
	test that build() makes directories and copy drz and flt data. 
	"""
	b = Batch(dir_data+'roadmap_lite.json', directory=dir_testing)
	b.build()
	assert os.path.isfile(b.directory+'SDSSJ0832+1615/obs0/drz0/drz.fits')
	assert os.path.isfile(b.directory+'SDSSJ0832+1615/obs0/drz0/flt0/flt.fits')


class Count():
	def __init__(self):
		self.count = 0


def test_batch_iterdrz():
	b = Batch(dir_data+'roadmap_lite.json', directory=dir_testing)
	b.build()
	c = Count()

	def iterfunc(drz, obs, tar, c):
		c.count += 1

	b.iterdrz(iterfunc, c=c)
	assert c.count == 4


def test_batch_iterflt():
	b = Batch(dir_data+'roadmap_lite.json', directory=dir_testing)
	b.build()
	c = Count()

	def iterfunc(flt, drz, obs, tar, c):
		c.count += 1

	b.iterflt(iterfunc, c=c)
	assert c.count == 24


def test_batch_make_roadmap():
	"""
	make roadmap from tables. 
	"""
	dir_tab = dir_data+'table/'
	fp_roadmap = dir_testing+'roadmap_made.json'
	status = batch.make_roadmap(fp=fp_roadmap, 
						fp_tab_target=dir_tab+'target.csv',
						fp_tab_observations=[dir_tab+'obs0.csv', dir_tab+'obs1.csv'],
						fp_tab_sources=[dir_tab+'source_qso.csv',dir_tab+'source_star0.csv',dir_tab+'source_star1.csv',],
						source_names=['qso', 'star0', 'star1'],
						dir_data='/Users/aisun/Documents/astro/projects/feedback/followup/hst/zakamska_erq/raw_data/hst/everything/')

	assert status
	assert os.path.isfile(fp_roadmap)
	with open(fp_roadmap, 'r') as f:
		r = json.load(f)

	assert len(r[0]["sources"]) == 3
	assert r[0]["observations"][0]["drzs"][0]["flts"][0]["sources"]["qso"]["x"] == 556
	assert r[0]["observations"][0]["drzs"][0]["flts"][0]["sources"]["qso"]["spectrum_form"] == "powerlaw_nu"

	# drz and flt should have different qso x, y location
	assert r[0]["observations"][0]["drzs"][0]["flts"][0]["sources"]["qso"]["x"] != r[0]["observations"][0]["drzs"][0]["sources"]["qso"]["x"]



def test_batch_compile_drz():
	"""
	test that build() makes directories and copy drz and flt data. 
	"""
	b = Batch(dir_data+'roadmap_lite.json', directory=dir_testing)
	b.build()

	def funcdrz_make_number_csv(drz, obs, tar, source='qso'):
		s = drz.sources[source]
		df = pd.DataFrame()
		df['number_1'] = [random.randint(0, 10)]
		df['number_2'] = [random.randint(10, 20)]
		if not os.path.isdir(s.directory):
			os.mkdir(s.directory)
		df.to_csv(s.directory+'number.csv', index=False)

	b.iterdrz(funcdrz_make_number_csv, source='qso')
	assert os.path.isfile(b.targets[0].observations[0].drzs[0].sources['qso'].directory+'number.csv')

	fp_out = dir_testing+'compiled_number.csv'
	dfnumber = b.compiledrz_source(fn='number.csv', fp_out=fp_out, source='qso')

	assert os.path.isfile(fp_out)

	dfnumber_read = pd.read_csv(fp_out)
	assert len(dfnumber_read) == 4
	assert dfnumber_read.loc[0, 'target'] == 'SDSSJ0832+1615'
	assert dfnumber_read.loc[0, 'drz'] == 'drz0'
	assert dfnumber_read.loc[0, 'source'] == 'qso'


def test_batch_merge_drz():

	b = Batch(dir_data+'roadmap_two_drzs.json', directory=dir_testing)
	assert len(b.targets[0].observations[0].drzs) == 2
	assert len(b.targets[0].observations[0].drzs[0].flts) == 4
	assert len(b.targets[0].observations[0].drzs[1].flts) == 2

	b.merge_drzs()
	assert len(b.targets[0].observations[0].drzs) == 1
	drz_master = b.targets[0].observations[0].drzs[0]
	assert len(drz_master.flts) == 6
	assert drz_master.name == 'drz_master'
	for i in range(len(drz_master.flts)):
		drz_master.flts[i].directory = drz_master.directory+'flt{}/'.format(str(i))

