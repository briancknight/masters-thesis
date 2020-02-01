from imTools import *
from scipy.spatial import ConvexHull
from convexHullPlot import plotLowerHull
import cvxpy as cvx


def Taylor08(target, base, window):
	"""target, deformed, m x n x 3 RGB images"""
	(m,n, _ ) = target.shape

	px = np.random.random(19)
	py = np.random.random(19)

	window = 10
	k = 4 # 16 kernels
	sigma = 10

	(Ax, Ay, Iz, b, Dx, Dy) = getConstraintCoeffs(target, base, window, k, sigma, px, py)

	return (Ax, Ay, Iz, b, Dx, Dy)


def getConstraintCoeffs(target, base, window, k, sigma, px, py):

	(m, n, _) = target.shape
	# initializing coefficient matrices & vectors
	Ax = []
	Ay = []
	Iz = []
	b = []
	Dx = []
	Dy = []

	kernels = getKernels((m, n), k)

	for x in range(m):
		for y in range(n):
			dx = gaussianD((x,y), kernels, sigma, px)
			dy = gaussianD((x,y), kernels, sigma, py)

			if ((x + dx)*(y + dy) > 0) and (dx + x < m) and (dy + y< n):

				errorSurface = []

				for i in range(-window, window):
					for j in range(-window, window):
						try:

							error = np.linalg.norm(target[x,y] - base[x + dx + i, y + dy + j], 3)
							errorSurface.append([x + dx + i, y + dy + j, error])

						except: IndexError

				if len(errorSurface) > 4:
					errorSurface = np.array(errorSurface)

					hull = ConvexHull(points = errorSurface)

					Ax1 = []
					Ay1 = []
					Iz1 = []
					for i in range(len(hull.simplices)):

						if hull.equations[i][2] < 0:
							ax = hull.equations[i][0]
							ay = hull.equations[i][1]
							az = hull.equations[i][2]
							dist = hull.equations[i][3]
							
							Ax1.append(ax)
							Ay1.append(ay)
							Iz1.append(az)
							b.append(dist)

					Ax.append(Ax1)
					Ay.append(Ay1)
					Iz.append(Iz1)
					Dx.append(dx)
					Dy.append(dy)

	Ax = coeffMatFormat(Ax)
	Ay = coeffMatFormat(Ay)
	Iz = coeffMatFormat(Iz)
	b = np.array(b)
	Dx = np.array(Dx)
	Dy = np.array(Dy)

	return (Ax, Ay, Iz, b, Dx, Dy)


def coeffMatFormat(mat):
	"""Converts coefficients to form in paper"""
	M = len(mat)
	S = 0
	for a in mat:
		S+=len(a)
	
	A = np.zeros([S, M])

	j = 0
	for i in range(len(mat)):
	    a = mat[i]
	    l = len(a)
	    A[j:j+l, i] = a
	    j+=l

	return np.array(A)


def getKernels(dims, k):
	"""
	im should be a numpy array, this returns evenly
	distributed seed points for segmentation algorithms

	returns n^2 seeds, evently distributed in im
	"""
	(m, n) = dims

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

def secondOrderD(pix, p):

	(x,y) = pix
	C = np.array([1, x, y, x*y, x**2, y**2])
	# D = np.dot(C, p)

	return int(round(np.dot(C, p)))

def gaussianD(pix, kernels, sigma, p):

	(x, y) = pix

	C = [1, x, y]

	for kernel in kernels:
		r = np.linalg.norm(np.array([x, y]) -  np.array(kernel), 2)
		C.append(np.exp(-r/(sigma**2)))

	# D = np.dot(C, p)

	return int(round(np.dot(C, p)))


def secondOrderDeformImage(im, px, py):

	(m,n, _) = base.shape

	defomredIm = np.zeros([m, n, 3])

	def secondOrderD(pix, p):

		(x,y) = pix
		C = np.array([1, x, y, x*y, x**2, y**2])
		# D = np.dot(C, p)

		return int(round(np.dot(C, p)))

	for i in range(m):
		for j in range(n):
			# (Dx, Dy) = firstOrderD((i,j), px, py)
			Dx = secondOrderD((i, j), px)
			Dy = secondOrderD((i, j), py)

			try:
				deformedIm[x + Dx, y + Dy] = im[i,j]
			except: IndexError

	return deformedIm.astype(int)


def gaussianDeformImage(im, sigma, k, px, py):

	(m,n, _) = im.shape

	kernels = getKernels((m,n), k)

	deformedIm = np.zeros([m, n, 3])

	for i in range(m):
		for j in range(n):

			Dx = gaussianD((i, j), kernels, sigma, px)
			Dy = gaussianD((i, j), kernels, sigma, py)

			try:
				deformedIm[i + Dx, j + Dy] = im[i,j]

			except: IndexError

	return deformedIm.astype(int)



def main():

	target = readImage('images/BrainProtonDensitySliceShifted13x17y.png', (100,100))
	base = readImage('images/BrainT1Slice.png', (100,100))
	(m, n, _) = base.shape

	px = np.random.random(19)
	py = np.random.random(19)

	plt.imshow(gaussianDeformImage(base, target, px, py))
	plt.show()

	Taylor08(target, base, 10)
	# (A, B) = getConstraints(target, base, 5, 4, px, py)

	# print(A[1])
	# print("\n\n\n\n")
	# print(B)

if __name__ == '__main__':
	main()