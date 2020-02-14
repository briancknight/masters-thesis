from imTools import *
from scipy.spatial import ConvexHull
from convexHullPlot import plotLowerHull
from deformationFunctions import *
import cvxpy as cvx


def Taylor08(target, base, window, px, py):
	"""target, deformed, m x n x 3 RGB images"""
	(m,n, _ ) = target.shape

	# px = np.random.random(19)
	# py = np.random.random(19)

	window = 10
	k = 4 # 16 kernels
	sigma = 10

	(Ax, Ay, Iz, b, Dx, Dy, C) = getConstraintCoeffs(target, base, window, k, sigma, px, py)

	deformedIm = gaussianDeformImage(target, sigma, k, px, py)

	return (deformedIm, Ax, Ay, Iz, b, Dx, Dy, C)


def getConstraintCoeffs(target, base, window, k, sigma, px, py):
	# Currently Structured for a Gaussian Deformation Model

	(m, n, _) = target.shape
	# initializing coefficient matrices & vectors
	Ax = []
	Ay = []
	Iz = []
	b = []
	Dx = []
	Dy = []
	C = []
	kernels = getKernels((m, n), k)

	for x in range(m):
		for y in range(n):
			# Get deformation basis function values
			c = gaussianD((x,y), kernels, sigma)
			# Calculate deformation at x, y
			dx = int(round(np.dot(c, px)))
			dy = int(round(np.dot(c, py)))

			# Check if within base image range:
			if ((x + dx)*(y + dy) > 0) and (dx + x < m) and (dy + y< n):

				# Contructs Error Surface
				errorSurface = []

				for i in range(-window, window):
					for j in range(-window, window):
						try:

							error = np.linalg.norm(target[x,y] - base[x + dx + i, y + dy + j], 3)
							errorSurface.append([x + dx + i, y + dy + j, error])

						except: IndexError

				if len(errorSurface) > 4: # Check we have enough points
					errorSurface = np.array(errorSurface)

					hull = ConvexHull(points = errorSurface)

					Ax1 = []
					Ay1 = []
					Iz1 = []
					# Get lower planar facet coefficients
					for i in range(len(hull.simplices)):

						if hull.equations[i][2] < 0:
							ax = hull.equations[i][0]
							ay = hull.equations[i][1]
							az = hull.equations[i][2]
							dist = hull.equations[i][3]
							
							Ax1.append(ax)
							Ay1.append(ay)
							Iz1.append(1)
							b.append(dist)

					Ax.append(Ax1)
					Ay.append(Ay1)
					Iz.append(Iz1)
					Dx.append(dx)
					Dy.append(dy)
					C.append(c)

	Ax = coeffMatFormat(Ax)
	Ay = coeffMatFormat(Ay)
	Iz = coeffMatFormat(Iz)
	b = np.array(b)
	Dx = np.array(Dx)
	Dy = np.array(Dy)
	C = np.array(C)

	return (Ax, Ay, Iz, b, Dx, Dy, C)


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


def getHessians(Ax, Ay, C, s):

	# Work in progress... 4 more D_i's to go!
	
	D1 = Ax.T@np.diag(1/s**2)@Ax
	D2 = Ay.T@np.diag(1/s**2)@Ay

	Hpx = C.T@D1@C
	Hpy = C.T@D1@C

	HalfHp = np.concat((Hpx, Hpy), axis = 1)

	return HalfHp




def main():

	target = readImage('images/BrainProtonDensitySliceShifted13x17y.png', (100,100))
	base = readImage('images/BrainT1Slice.png', (100,100))
	(m, n, _) = base.shape

	px = np.random.random(19)
	py = np.random.random(19)

	plt.imshow(gaussianDeformImage(base, target, px, py))
	plt.show()

	Taylor08(target, base, 10, px, py)

if __name__ == '__main__':
	main()