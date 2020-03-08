import numpy as np

def secondOrderD(pix):

	(x,y) = pix
	# C = np.array([1, x, y, x*y, x**2, y**2])
	return np.array([1, x, y, x*y, x**2, y**2])

def firstOrderD(pix):

	(x,y) = pix

	return np.array([1, x, y])

def gaussianD(pix, kernels, sigma):

	(x, y) = pix

	C = [1, x, y]

	for kernel in kernels:
		r = np.linalg.norm(np.array([x, y]) -  np.array(kernel), 2)
		C.append(np.exp(-r/(sigma**2)))

	# return int(round(np.dot(C, p)))
	return C


def secondOrderDeformImage(im, px, py):

	(m,n, _) = im.shape

	deformedIm = np.zeros([m, n, 3])

	for i in range(m):
		for j in range(n):
			c = secondOrderD((i,j))
			Dx = int(round(np.dot(c, px)))
			Dy = int(round(np.dot(c, py)))


			if (i + Dx < m) and (j + Dy) < n: 
				if (i + Dx >= 0) and (j + Dy >= 0):

					deformedIm[i + Dx, j + Dy] = im[i,j]


	return deformedIm.astype(int)

def firstOrderDeformImage(im, px, py):

	(m, n, _) = im.shape

	deformedIm = np.zeros([m, n, 3])

	for i in range(m):
		for j in range(n):
			c = firstOrderD((i,j))
			Dx = int(round(np.dot(c, px)))
			Dy = int(round(np.dot(c, py)))

			if (i + Dx < m) and (j + Dy) < n: 
				if (i + Dx >= 0) and (j + Dy >= 0):
					
					deformedIm[i + Dx, j + Dy] = im[i,j]

	return deformedIm.astype(int)

def xyShift(im, dx, dy):

	(m, n, _) = im.shape

	deformedIm = np.zeros([m, n, 3])

	for i in range(m):
		for j in range(n):
			if (i + Dx < m) and (j + Dy) < n and (i + Dx >= 0) and (j + Dy >= 0): 
						
				deformedIm[i + Dx, j + Dy] = im[i,j]

def gaussianDeformImage(im, sigma, k, px, py):

	(m,n, _) = im.shape

	kernels = getKernels((m,n), k)

	deformedIm = np.zeros([m, n, 3])

	for i in range(m):
		for j in range(n):

			c = gaussianD((i, j), kernels, sigma)
			Dx = int(round(np.dot(c, px)))
			Dy = int(round(np.dot(c, py)))


			if (i + Dx < m) and (j + Dy) < n: 
				if (i + Dx >= 0) and (j + Dy >= 0):
					
					deformedIm[i + Dx, j + Dy] = im[i,j]

	return deformedIm.astype(int)
	

def getKernels(dims, k):
	"""
	im should be a numpy array, this returns evenly
	distributed seed points for segmentation algorithms

	returns k^2 seeds, evently distributed in im
	"""
	(n, m) = dims

	if k**2 > max(m, n):
		raise Exception('Image too small for this many kernels')

	kernels = []

	if k == 1:
		kernel = (int(np.floor(m / 2)), int(np.floor(n / 2)))
		kernels.append(kernel)

		return kernels

	else:
		xStart = np.floor(m / k)
		xEnd = m - xStart

		yStart = np.floor(n / k)
		yEnd = n - yStart

		x = np.linspace(xStart, xEnd, k).astype(int)
		y = np.linspace(yStart, yEnd, k).astype(int)

		grid, _ = np.meshgrid(x, y)

		for i in range(k):
			for j in range(k):

				kernels.append((grid[0][i], grid[0][j]))

		return kernels
