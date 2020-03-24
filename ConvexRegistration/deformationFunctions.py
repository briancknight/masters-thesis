import numpy as np
from Transforms import displacementTransform

def secondOrderD(pix):

	(x,y) = pix
	# C = np.array([1, x, y, x*y, x**2, y**2])
	return np.array([1, x, y, x*y, x**2, y**2])

def firstOrderD(pix):

	(x,y) = pix

	return np.array([1, x, y])

def gaussianD(pix, kernels, sigma):

	(x,y) = pix

	C = [1, x, y]

	for kernel in kernels:
		r = np.linalg.norm(np.array([x, y]) -  np.array(kernel), 2)
		C.append(np.exp(-r/(sigma**2)))

	# return int(round(np.dot(C, p)))
	return C


def secondOrderDeformImage(im, p):

    if len(p) != 12:
        raise Exception('Incorrect number of deformation parameters for Second Order Deformation')

    px = p[:6]
    py = p[6:]

    (rows, cols, _) = im.shape

    displacements = np.zeros([rows, cols, 2])

    for y in range(rows):
        for x in range(cols):

            c = secondOrderD((x, y))
            Dx = np.dot(c, px)
            Dy = np.dot(c, py)
            # The minuse defines our orientation (down and right is  ++)
            displacements[y, x] = [-Dx, -Dy]

    return displacementTransform(im, displacements)


def firstOrderDeformImage(im, p):

	# Models and affine transformation

	if len(p) != 6:
	    raise Exception('Incorrect number of deformation parameters for Second Order Deformation')

	px = p[:3]
	py = p[3:]

	(rows, cols, _) = im.shape

	displacements = np.zeros([rows, cols, 2])

	for y in range(rows):
		for x in range(cols):

			c = firstOrderD((x,y))
			Dx = np.dot(c, px)
			Dy = np.dot(c, py)
			displacements[y, x] = [-Dx, -Dy]

	return displacementTransform(im, displacements)


def gaussianDeformImage(im, sigma, k, p):

	if len(p) != 38:
	    raise Exception('Incorrect number of deformation parameters for Second Order Deformation')

	px = p[:19]
	py = p[19:]

	(rows,cols, _) = im.shape

	kernels = getKernels((rows, cols), k)

	displacements = np.zeros([rows, cols, 2])

	for y in range(rows):
		for x in range(cols):

			c = gaussianD((x,y), kernels, sigma)
			Dx = np.dot(c, px)
			Dy = np.dot(c, py)
			displacements[y, x] = [-Dx, -Dy]

	return displacementTransform(im, displacements)
	

def getKernels(dims, k):
	"""
	im should be a numpy array, this returns evenly
	distributed seed points for segmentation algorithms

	returns k^2 seeds, evently distributed in im
	"""
	(rows, cols) = dims

	if k**2 > max(rows, cols):
		raise Exception('Image too small for this many kernels')

	kernels = []

	if k == 1:
		kernel = (int(np.floor(rows / 2)), int(np.floor(cols / 2)))
		kernels.append(kernel)

		return kernels

	else:
		xStart = np.floor(cols / k)
		xEnd = cols - xStart

		yStart = np.floor(rows / k)
		yEnd = rows - yStart

		x = np.linspace(xStart, xEnd, k).astype(int)
		y = np.linspace(yStart, yEnd, k).astype(int)

		grid, _ = np.meshgrid(x, y)

		for i in range(k):
			for j in range(k):

				kernels.append((grid[0][i], grid[0][j]))

		return kernels
