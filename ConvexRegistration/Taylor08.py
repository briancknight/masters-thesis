from imTools import *
from scipy.spatial import ConvexHull
from scipy import sparse
from convexHullPlot import plotLowerHull
from deformationFunctions import *
import cvxpy as cvx
import time


def Taylor08(target, base, window, px, py):
	"""Inputs: 
				TARGET, BASE: m x n x 3 RGB images
				WINDOW: integer defining the size 
				of the window for convex approximation
				PX, PY: vectors of length 3 + # of kernels
				Z: auxilary variable to be minimized

	"""
	(m,n, _ ) = target.shape

	k = 4 # 16 kernels
	sigma = 10

	(deformedIm, Ax, Ay, Iz, b, Dx, Dy, C) = getConstraintCoeffs(target, base, window, k, sigma, px, py)

	# A = (Ax @ C @ px) + (Ay @ C @ py) - z

	# s = b - A
	# d = sparse.diags(s**(-1))
	# d2 = d**2

	# (Hp, Hz, D6) = getHessians(Ax, Ay, Iz, C, d2)

	return (deformedIm, Ax, Ay, Iz, b, Dx, Dy, C)


# Marginally improved.
def getConstraintCoeffs(target, base, window, k, sigma, px, py):
	# Currently Structured for a Gaussian Deformation Model
	start = time.time()
	print("Getting Coefficients...")
	# deformedIm = gaussianDeformImage(base, sigma, k, px, py)
	deformedIm = firstOrderDeformImage(base, px, py)
	(m, n, _) = target.shape
	# initializing coefficient matrices & vectors
	Ax = np.array([])
	Ay = np.array([])
	Iz = np.array([])
	b = []
	Dx = []
	Dy = []
	C = []
	numFacets = []
	kernels = getKernels((m, n), k)

	for x in range(m):
		for y in range(n):
			# Get deformation basis function values
			# c = gaussianD((x,y), kernels, sigma)
			c = firstOrderD((x,y))
			# Calculate deformation at x, y
			dx = int(round(np.dot(c, px)))
			dy = int(round(np.dot(c, py)))

			# Contructs Error Surface
			errorSurface = []

			""" for some predefined window, construct lower convex hull
			of error function between target[x,y] and deformedIm[x,y]
			"""
			for i in range(-window, window):
				for j in range(-window, window):

					if ((x + i) < m and (x + i) >= 0 and (y+j) < n and (y+j) >=0):
						error = np.linalg.norm(target[x,y] - deformedIm[x + i, y + j], 3)
					else: 
						error = np.linalg.norm(target[x,y], 3)

					# except: IndexError

					errorSurface.append([x + i, y + j, error])


			hull = ConvexHull(points = errorSurface, qhull_options='QJ')
			# hull = ConvexHull(points = errorSurface)
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

			numFacets.append(len(Ax1))

			# Ax.append(Ax1)
			# Ay.append(Ay1)
			# Iz.append(Iz1)
			Ax = np.append(Ax, Ax1)
			Ay = np.append(Ay, Ay1)
			Iz = np.append(Iz, Iz1)
			# Ax = np.concatenate((Ax, Ax1), axis = 0)
			# Ay = np.concatenate((Ay, Ay1), axis = 0)
			# Iz = np.concatenate((Iz, Iz1), axis = 0)
			Dx.append(dx)
			Dy.append(dy)
			C.append(c)

	print("Done. Time elapsed:", time.time() - start, " \n\n")

	formatT = time.time()
	print("Formatting Matrices...")
	
	Ax = coeffMatFormat(Ax, numFacets)
	Ay = coeffMatFormat(Ay, numFacets)
	Iz = coeffMatFormat(Iz, numFacets)
	b = np.array(b)
	Dx = np.array(Dx)
	Dy = np.array(Dy)
	C = np.array(C)

	print("Done. Total Time Elapsed: ", time.time() - start, "\n\n")

	return (deformedIm, Ax, Ay, Iz, b, Dx, Dy, C)


def coeffMatFormatOLD(mat):
	"""Converts coefficients to form in paper
	Update: THIS IS SO SLOW"""
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

	return sparse.csc_matrix(np.array(A))

def coeffMatFormat(A, numFacets):
    """Converts coefficients to form in paper
    Much faster :) """
    col = np.array([])
    for i in range(len(numFacets)):
        col = np.concatenate((col, [i]*numFacets[i]), axis = 0)
        
    row = np.array([i for i in range(len(A))])
    
    return sparse.csc_matrix((A, (row, col)), shape = (len(A), len(numFacets)))


def getHessians(Ax, Ay, Iz, C, d):

	
	D1 = Ax.T @ d @ Ax
	D2 = Ay.T @ d @ Ax
	D3 = Ay.T @ d @ Ay
	D4 = Iz.T @ d @ Ax
	D5 = Iz.T @ d @ Ay
	D6 = Iz.T @ d @ Iz

	# Constructs Hp
	Hpx   = C.T @ D1 @ C
	Hpxpy = C.T @ D2 @ C
	Hpy   = C.T @ D3 @ C

	Hp1 = np.concatenate((Hpx, Hpxpy), axis = 1)
	Hp2 = np.concatenate((Hpxpy, Hpy), axis = 1)
	Hp  = np.concatenate((Hp1, Hp2),   axis = 0)

	# Constructs Hz
	Hz1 = D4 @ C
	Hz2 = D5 @ C
	Hz  = np.concatenate((Hz1, Hz2), axis = 1)

	return (Hp, Hz, D6)


def TaylorNewtonStep(Ax, Ay, Iz, b, s, d, d2, p, z):    
	""" 
	*CURRENTLY BROKEN**

	With each iteration we are seeking to minimize 
	t 1.T@z - sum[ log(b - AxCpx - (Ay @ C @ py)  + (Iz @ z)) ]

	"""
	# parameters for line search:
	alpha = 0.01
	beta = 0.8
	eps = 1/(10**6)
	maxiters = 100
	count = 0
	L = Ax.shape[1]

	#     gpx = -1 * C.T@Ax.T@s**(-1)
	#     gpy = -1 * C.T@Ay.T@s**(-1)
	#     gp = np.concatenate((gpx, gpy), axis = 0)
	for iter in range(maxiters):
		count += 1
		#(Hp, Hz) = getHessians(Ax, Ay, Iz, C, np.diag(1/s**(2)))
		(Hp, Hz, D6) = getHessians(Ax, Ay, Iz, C, d2)

		gpx = -1 * C.T@Ax.T@s**(-1)
		gpy = -1 * C.T@Ay.T@s**(-1)
		gp = np.concatenate((gpx, gpy), axis = 0)
		gz = -1 * Iz.T@s**(-1)
		g = np.concatenate((gp, gz), axis = 0)
		Dinv = sparse.linalg.inv(D6)

		dp = np.linalg.solve(Hp - Hz.T @ Dinv @ Hz, gp - Hz.T @ Dinv @ gz)
		dz = sparse.linalg.inv(D6)@(gz - Hz @ dp)
		delta = np.concatenate((dp, dz), axis = 0)

		lambdasquared = -1*g.T @ delta

		# otherwise perform line search:
		t = 1

		while  (t * np.ones(len(dz)).T @ dz - np.sum( np.log( b - (Ax@C@(p + t * dp)[:19]) - (Ay @ C @ (p + t * dp)[19:])) + Iz@(z + dz)) - alpha * t * (gz.T @ deltaZ) > 0):
			t = beta * t
	    
		p = p + t * dp

		if count == maxiters:
			print('ERROR: MAXITERS reached.\n')
			p = 0

	return p

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