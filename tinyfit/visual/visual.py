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


	def plot(self, fn_out, vmin=None, vmax=None, cmap=plt.cm.viridis, stretch='linear', colorbar=False, figsize=(8, 6)):
		"""
		make plot of the entire fits image

		Args: 
			fn_out (str): output file name
		"""
		norm = get_norm(stretch=stretch, vmin=vmin, vmax=vmax)

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



	def biplot(self, fn, fn_out, vmin=None, vmax=None, cmap=plt.cm.viridis, stretch='linear', colorbar=False, figsize=(12, 6)):
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

		norm = get_norm(stretch=stretch, vmin=vmin, vmax=vmax)

		# plotting
		plt.close('all')
		fig = plt.figure(figsize=figsize)

		ax1 = fig.add_subplot(121, aspect='equal')
		ax1.axis('off')
		im = ax1.imshow(self.img, origin='lower', cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
		ax1.get_xaxis().set_visible(False)
		ax1.get_yaxis().set_visible(False)
		if colorbar:
			plt.colorbar(im, ax=ax1, orientation='horizontal', fraction=0.046, pad=0.04)

		ax2 = fig.add_subplot(122, aspect='equal')
		ax2.axis('off')
		im = ax2.imshow(img2, origin='lower', cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
		ax2.get_xaxis().set_visible(False)
		ax2.get_yaxis().set_visible(False)
		if colorbar:
			plt.colorbar(im, ax=ax2, orientation='horizontal', fraction=0.046, pad=0.04)

		plt.tight_layout()
		plt.savefig(fn_out)



	def ratioplot(self, fn, fn_out, vmin=None, vmax=None, cmap=plt.cm.viridis, stretch='linear', colorbar=False, figsize=(16, 6)):
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

		norm = get_norm(stretch=stretch, vmin=vmin, vmax=vmax)

		# plotting
		plt.close('all')
		fig = plt.figure(figsize=figsize)

		ax1 = fig.add_subplot(131, aspect='equal')
		ax1.axis('off')
		im = ax1.imshow(self.img, origin='lower', cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
		ax1.get_xaxis().set_visible(False)
		ax1.get_yaxis().set_visible(False)
		ax1.set_title('a')
		if colorbar:
			plt.colorbar(im, ax=ax1, orientation='horizontal', fraction=0.046, pad=0.04)

		ax2 = fig.add_subplot(132, aspect='equal')
		ax2.axis('off')
		im = ax2.imshow(img2, origin='lower', cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
		ax2.get_xaxis().set_visible(False)
		ax2.get_yaxis().set_visible(False)
		ax2.set_title('b')
		if colorbar:
			plt.colorbar(im, ax=ax2, orientation='horizontal', fraction=0.046, pad=0.04)


		ax3 = fig.add_subplot(133, aspect='equal')

		ax3.axis('off')
		rimg = img2/np.absolute(self.img)
		norm = get_norm(stretch=stretch, vmin=-1., vmax=1.)
		im = ax3.imshow(rimg, origin='lower', cmap=cmap, norm=norm, vmin=-1., vmax=1.)
		ax3.get_xaxis().set_visible(False)
		ax3.get_yaxis().set_visible(False)
		ax3.set_title('b/abs(a)')
		if colorbar:
			plt.colorbar(im, ax=ax3, orientation='horizontal', fraction=0.046, pad=0.04)

		plt.tight_layout()
		plt.savefig(fn_out)


def get_norm(stretch='log', vmin=None, vmax=None):
	""" retur normalization function for plotting """
	if stretch == 'log':
		return DS9LogNorm(vmin=vmin, vmax=vmax)
	elif stretch == 'symlog': 
		return colors.SymLogNorm(linthresh=0.2, vmin=vmin, vmax=vmax)
	elif stretch == 'linear':
		return colors.Normalize(vmin=vmin, vmax=vmax)



class DS9LogNorm(colors.Normalize):
	"""
	Normalize a given value to the 0-1 range on a log scale
	https://aplpy.readthedocs.io/en/stable/normalize.html 
	"""
	def __call__(self, value, clip=None):
		raise Exception("log normalization not functioning")

		if clip is None:
			clip = self.clip

		result, is_scalar = self.process_value(value)

		result = np.ma.masked_less_equal(result, 0, copy=False)

		self.autoscale_None(result)
		vmin, vmax = self.vmin, self.vmax
		v0 = vmin - 0.1
		m = (vmax - v0)/(vmin - v0)

		if vmin > vmax:
			raise ValueError("minvalue must be less than or equal to maxvalue")
		elif vmin == vmax:
			result.fill(0)
		else:
			if clip:
				mask = np.ma.getmask(result)
				result = np.ma.array(np.clip(result.filled(vmax), vmin, vmax), mask=mask)
			# in-place equivalent of above can be much faster
			resdat = result.data
			mask = result.mask
			if mask is np.ma.nomask:
				mask = (resdat <= 0)
			else:
				mask |= resdat <= 0
			np.copyto(resdat, 1, where=mask)

			y = np.log10((resdat - vmin)/(vmax-vmin)*(m-1)+1)/np.log10(m)

			result = np.ma.array(y, mask=mask, copy=False)
		if is_scalar:
			result = result[0]
		return result

	def autoscale(self, A):
		"""
		Set *vmin*, *vmax* to min, max of *A*.
		"""
		A = np.ma.masked_less_equal(A, 0, copy=False)
		self.vmin = np.ma.min(A)
		self.vmax = np.ma.max(A)

	def autoscale_None(self, A):
		"""autoscale only None-valued vmin or vmax."""
		if self.vmin is not None and self.vmax is not None:
			return
		A = np.ma.masked_less_equal(A, 0, copy=False)
		if self.vmin is None and A.size:
			self.vmin = A.min()
		if self.vmax is None and A.size:
			self.vmax = A.max()


	def inverse(self, value):
		if not self.scaled():
			raise ValueError("Not invertible until scaled")

		vmin, vmax = self.vmin, self.vmax
		v0 = vmin - 0.1
		m = (vmax - v0)/(vmin - v0)

		val = np.ma.asarray(value)
		x = (10.**(np.log10(m)*val)-1)/(m-1)
		return x


# class DS9LogNormalize(colors.Normalize):
# 	""" 
# 	stretch input array x into log scale
# 	v0 is a parameter that has to be smaller than vmin
# 	Follows the formulation of aplpy
# 	https://aplpy.readthedocs.io/en/stable/normalize.html 
# 	"""


# 	def __init__(self, vmin=None, vmax=None, v0=None, clip=False):
# 		print("WARNING: LOG Normalization is not functioning")
# 		self.v0 = v0

# 		colors.Normalize.__init__(self, vmin, vmax, clip)


# 	def __call__(self, value, clip=None):

# 		if self.v0 is None:
# 			self.v0 = self.vmin - 1.e-3

# 		m = (self.vmax - self.v0)/(self.vmin - self.v0)
# 		y = np.log10((value - self.vmin)/(self.vmax-self.vmin)*(m-1)+1)/np.log10(m)
# 		return y
