from astropy.io import fits

def main():
	fn_in = 'id9ya9v8q_flt.fits'

	hdus = fits.open(fn_in)

	hdus_new = fits.PrimaryHDU(hdus[1].data, header=hdus[1].header)

	hdus_new.writeto('science_img.fits')


if __name__ == '__main__':
	main()