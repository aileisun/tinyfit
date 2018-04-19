import pytest
import os
import shutil
import json

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
	assert b.targets[0].observations[0].drzs[0].flts[0].sources['qso'].spectrum_type == -1.33441542369
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
						dir_data='/Users/aisun/Documents/astro/projects/feedback/followup/hst/zakamska_erq/data/hst/everything/',
						dir_local=dir_testing
						)

	assert status
	assert os.path.isfile(fp_roadmap)
	with open(fp_roadmap, 'r') as f:
		r = json.load(f)

	assert r[0]["observations"][0]["drzs"][0]["flts"][0]["sources"]["qso"]["x"] == 556
	assert r[0]["observations"][0]["drzs"][0]["flts"][0]["sources"]["qso"]["spectrum_form"] == "powerlaw_nu"
