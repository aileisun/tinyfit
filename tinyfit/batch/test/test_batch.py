import pytest
import os
import shutil

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

	yield
	# tear down
	if os.path.isdir(dir_testing):
		shutil.rmtree(dir_testing)

def test_batch_init():
	b = Batch(roadmap=dir_data+'roadmap.json', directory=dir_testing)
	assert b.directory == dir_testing
	assert len(b.targets) == 2

	assert b.targets[0].name == "SDSSJ0832+1615"
	assert b.targets[0].directory == b.directory+"SDSSJ0832+1615/"
	assert b.targets[0].observations[0].directory == b.directory+"SDSSJ0832+1615/obs0/"
	assert b.targets[0].observations[0].drzs[0].directory == b.directory+"SDSSJ0832+1615/obs0/drz0/"
	assert b.targets[0].observations[0].drzs[0].flts[0].directory == b.directory+"SDSSJ0832+1615/obs0/drz0/flt0/"

	assert b.targets[0].observations[0].drzs[0].flts[0].sources[0].name == 'qso'
	assert b.targets[0].observations[0].drzs[0].flts[0].sources[0].x == 556
	assert len(b.targets[0].sources) == len(b.targets[0].observations[0].drzs[0].flts[0].sources)


def test_batch_run():
	pass