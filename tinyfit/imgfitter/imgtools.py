""" Contain small functions for imgfitter """

def get_cutout_xy_range(xc, yc, nx, ny):
	"""
	get cutout range x0, x1, y0, y1 given xc, yc, nx, ny
	"""
	xc, yc = int(xc), int(yc)

	if nx%2 == 0:
		x0, x1 = xc-nx//2, xc+nx//2
	elif nx%2 == 1:
		x0, x1 = xc-nx//2, xc+nx//2 + 1

	if ny%2 == 0:
		y0, y1 = yc-ny//2, yc+ny//2
	elif ny%2 == 1:
		y0, y1 = yc-ny//2, yc+ny//2 + 1

	return x0, x1, y0, y1