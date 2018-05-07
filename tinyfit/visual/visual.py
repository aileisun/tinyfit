"""
for plotting fits images
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.wcs as aw
import matplotlib.colors as colors
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


class visual(object):
	def __init__(self, fn, obj=None):
		"""
		make plots for fits images and handles wcs coordinates. 

		WARNING: The image read from fits file is in (y, x), i.e., x is the second coordinate and y is the first coordinate. 

		Args: 
			fn (str): path to the fits file
			obj (obsobj instance): that contains attributes, ra, dec, z, name

		"""
		self.img, self.header = fits.getdata(fn, header=True)
		self.wcs = aw.WCS(self.header)
		self.obj = obj


	def plot(self, fn_out, vmin=None, vmax=None, cmap=plt.cm.viridis, stretch='log', colorbar=False, figsize=(8, 6)):
		"""
		make plot of the entire fits image

		Args: 
			fn_out (str): output file name
		"""

		if stretch == 'symlog': 
			norm = colors.SymLogNorm(linthresh=0.2, vmin=vmin, vmax=vmax)
		elif stretch == 'linear':
			norm = colors.Normalize(vmin=vmin, vmax=vmax)

		# plotting
		plt.close('all')
		fig = plt.figure(figsize=figsize)

		ax = fig.add_subplot(111, aspect='equal')
		ax.axis('off')

		im = ax.imshow(self.img, origin='lower', cmap=cmap, norm=norm)

		if colorbar:
			plt.colorbar(im)
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		plt.tight_layout()
		plt.savefig(fn_out)
		plt.close()



	def biplot(self, fn, fn_out, vmin=None, vmax=None, cmap=plt.cm.viridis, stretch='log', colorbar=False, figsize=(12, 6)):
		"""
		plot two fits files side by side with the same color stretch. 

		Args: 
			fn (str): file name of the second input image
			fn_out (str): output file name
		"""

		img2, header2 = fits.getdata(fn, header=True)

		# set color stretch
		if vmin is None:
			vmin = min(self.img.min(), img2.min())

		if vmax is None:
			vmax = max(self.img.max(), img2.max())


		if stretch == 'symlog': 
			norm = colors.SymLogNorm(linthresh=0.2, vmin=vmin, vmax=vmax)
		elif stretch == 'linear':
			norm = colors.Normalize(vmin=vmin, vmax=vmax)

		# plotting
		plt.close('all')
		fig = plt.figure(figsize=figsize)

		ax1 = fig.add_subplot(121, aspect='equal')
		ax1.axis('off')

		im = ax1.imshow(self.img, origin='lower', cmap=cmap, norm=norm)

		ax1.get_xaxis().set_visible(False)
		ax1.get_yaxis().set_visible(False)
		if colorbar:
			plt.colorbar(im, ax=ax1, orientation='horizontal', fraction=0.046, pad=0.04)

		ax2 = fig.add_subplot(122, aspect='equal')
		ax2.axis('off')

		im = ax2.imshow(img2, origin='lower', cmap=cmap, norm=norm)

		ax2.get_xaxis().set_visible(False)
		ax2.get_yaxis().set_visible(False)

		if colorbar:
			plt.colorbar(im, ax=ax2, orientation='horizontal', fraction=0.046, pad=0.04)


		plt.tight_layout()
		plt.savefig(fn_out)
		plt.close()


def get_norm(stretch='log', vmin=None, vmax=None):

	if stretch == 'log':
		norm = DS9LogNormalize(vmin=vmin, vmax=vmax)
	elif stretch == 'symlog': 
		norm = colors.SymLogNorm(linthresh=0.2, vmin=vmin, vmax=vmax)
	elif stretch == 'linear':
		norm = colors.Normalize(vmin=vmin, vmax=vmax)


class DS9LogNormalize(colors.Normalize):
	""" 
	stretch input array x into log scale
	v0 is a parameter that has to be smaller than vmin
	Follows the formulation of aplpy
	https://aplpy.readthedocs.io/en/stable/normalize.html 
	"""

	def __init__(self, vmin=None, vmax=None, v0=None, clip=False):

		self.v0 = v0

		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):

		if self.v0 is None:
			self.v0 = self.vmin - 1.e-3

		m = (self.vmax - self.v0)/(self.vmin - self.v0)
		x_norm = (value - self.vmin)/(self.vmax-self.vmin)
		y = np.log10(x_norm*(m-1)+1)/np.log(m)
		return y
