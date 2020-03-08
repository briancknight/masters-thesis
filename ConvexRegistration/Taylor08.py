from imTools import *
from scipy.spatial import ConvexHull
from scipy import sparse
from convexHullPlot import plotLowerHull
from deformationFunctions import *
import cvxpy as cvx
import time



def Taylor08(target, base):

	# Given feasible starring point z for p = zeros

	(Ax, Ay, Iz, b, C) = getConstraintCoeffs(target, base, 15, 4 ,10)

	z = np.ones(Iz.shape[1])*180
	t = 1
	mu = 20
	eps = 1/10

	M = len(b)
	p = np.zeros(38)

	while True:
		(pStar, zStar) = TaylorNewtonStep((Ax, Ay, Iz, b, C), p, z, t)
		p = pStar
		z = zStar
		gap = M/t
		print('Current gap:', gap, '\n\n')
		if gap <= eps:
			break
		t = mu*t

	return p


def TaylorNewtonStep(facetCoeffs, p, z, t):
    # minimizes t*z - sum(log(ax_i ci.T px + ay_i ci.T py - Izi.T z - bi)) wrt z, px, py
    (Ax, Ay, Iz, b, C) = facetCoeffs

    c = t*np.ones(Iz.shape[1])
    
    L = int(len(p) / 2)
    
    # line search params:
    alpha = 0.3
    beta = 0.8
    eps =  1/10**6 # Not sure what to make this
    maxiters = 100
    count = 0
    
    # Perform Line Search:
    for iter in range(maxiters):
        
        count += 1
        A = (Ax @ C @ p[:L]) + (Ay @ C @ p[L:]) - Iz @ z
        s = b - A
        d = sparse.diags(s**(-1))
        d2 = d**2

        (Hp, Hz, D6) = getHessians(Ax, Ay, Iz, C, d2)
        (gp, gz) = getGradients(Ax, Ay, Iz, C, s, t)
        
        # Solve for Newton Step:
        g = np.concatenate((gp,gz), axis = 0)
        
        Dinv = sparse.linalg.inv(D6)
        Hprime = Hp - Hz.T @ Dinv @ Hz
        gpprime = (-1 * gp) - Hz.T @ Dinv @ (-1 * gz)
        
        # Use Block Diagonals and Schur Complement to solve the system:
        # dp = np.linalg.solve(Hp - Hz.T @ Dinv @ Hz, (-1*gp) - Hz.T @ Dinv @ (-1*gz)) 
        dp = np.linalg.solve(Hprime, gpprime)
        dz = sparse.linalg.inv(D6)@((-1*gz) - Hz @ dp)
        delta = np.concatenate((dp, dz), axis = 0)
#         gprime = gp - Hz.T @ Dinv @ gz
        
        g2 = np.concatenate((gpprime, gz), axis = 0)
                
        # Check Optimality Gap:
#         lambdasqr = -1 * dp.T @ Hprime @ dp
        lambdasqr = -1 * g.T @ delta
        # print('lambdasqr/2 = ', lambdasqr/2, '\n\n')
        if lambdasqr / 2 < eps:
            break # if already eps-suboptimal
       
        # else: 
        tau = 1        
        
        # Check if z + tau * dz is feasible
        while max(Ax @ C @ (p + tau * dp)[:L] + \
                  Ay @ C @ (p + tau * dp)[L:] - \
                  Iz @ (z + tau * dz) - b) >= 0.0:

            # Update tau
            tau = beta * tau

        # Want f(x + t*x_nt) < f(x) + t*alpha*g.T @ x_nt 
        while c.T @ (tau * dz) - sum(np.log(-1 * (Ax @ C @ (p + tau * dp)[:L] + \
                Ay @ C @ (p + tau * dp)[L:] - Iz @ (z + tau * dz) - b))) \
                + sum(np.log(-1 * (Ax @ C @ p[:L] + Ay @ C @ p[L:] - Iz @ z - b))) - \
                alpha * tau * g.T @ delta > 0:
            # Update tau
            tau = beta * tau
                    
        p += tau * dp
        z += tau * dz
        

    if count == maxiters:
        print('ERROR: MAXITERS reached.\n')
        p = 0
        z = 0
    
    return (p, z)


def getConstraintCoeffs(target, base, window, k, sigma):
	# Currently Structured for a Gaussian Deformation Model
	start = time.time()
	print("Getting Coefficients...")

	(m, n, _) = target.shape
	# initializing coefficient matrices & vectors
	Ax = np.array([])
	Ay = np.array([])
	Iz = np.array([])
	b = []
	C = []
	numFacets = []
	kernels = getKernels((m, n), k)

	for x in range(n):
		for y in range(m):

			# Contructs Error Surface
			errorSurface = []

			""" for some predefined window, construct lower convex hull
			of error function between target[x,y] and deformedIm[x,y]
			"""
			for i in range(-window, window):
				for j in range(-window, window):

					if ((x + i) < n and (x + i) >= 0 and (y+j) < m and (y+j) >=0):
						error = np.linalg.norm(target[x,y] - base[x + i, y + j], 3) ** 2

						# errorSurface.append([x + i, y + j, error])
						errorSurface.append([i, j, error])

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

			numFacets.append(len(Ax1))

			Ax = np.append(Ax, Ax1)
			Ay = np.append(Ay, Ay1)
			Iz = np.append(Iz, Iz1)
			C.append(gaussianD((x,y), kernels, sigma))

	print("Done. Time elapsed:", time.time() - start, " \n\n")

	formatT = time.time()
	print("Formatting Matrices...")
	
	Ax = coeffMatFormat(Ax, numFacets)
	Ay = coeffMatFormat(Ay, numFacets)
	Iz = coeffMatFormat(Iz, numFacets)
	b = np.array(b)

	C = np.array(C)

	print("Done. Total Time Elapsed: ", time.time() - start, "\n\n")

	return (Ax, Ay, Iz, b, C)


# Marginally improved.
def getConstraintCoeffs2(target, base, window, k, sigma, px, py):
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

	for x in range(n):
		for y in range(m):
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

					if ((x + i) < n and (x + i) >= 0 and (y+j) < m and (y+j) >=0):
						error = np.linalg.norm(target[x,y] - deformedIm[x + i, y + j], 3) ** 2

						# errorSurface.append([x + i, y + j, error])
						errorSurface.append([i, j, error])



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

			Ax = np.append(Ax, Ax1)
			Ay = np.append(Ay, Ay1)
			Iz = np.append(Iz, Iz1)
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


def coeffMatFormat(A, numFacets):
    """Converts coefficients to form in paper
    Much faster :) """
    col = np.array([])
    for i in range(len(numFacets)):
        col = np.concatenate((col, [i]*numFacets[i]), axis = 0)
        
    row = np.array([i for i in range(len(A))])
    
    return sparse.csc_matrix((A, (row, col)), shape = (len(A), len(numFacets)))


def getHessians(Ax, Ay, Iz, C, d):

	# Pls review Multivariate Calculus
	D1 = Ax.T @ d @ Ax
	D2 = Ax.T @ d @ Ay
	D3 = Ay.T @ d @ Ay
	D4 = -1 * Ax.T @ d @ Iz
	D5 = -1 * Ay.T @ d @ Iz
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

def getGradients(Ax, Ay, Iz, C, s, t):

    gz =  t - (Iz.T @ s**(-1))
    gpx =  C.T @ Ax.T @ s**(-1)
    gpy =  C.T @ Ay.T @ s**(-1)
    gp = np.concatenate((gpx, gpy), axis = 0)
    
    return (gp, gz)


def main():

	target = readImage('images/BrainT1SliceR10X13Y17.png', (50,50))
	base = readImage('images/BrainT1Slice.png', (50,50))
	(m, n, _) = base.shape

	p = Taylor08(base, target)

	im = gaussianDeformImage(base, 10, 4, p[:19], p[19:])

	plt.imshow(im)
	plt.show()


if __name__ == '__main__':
	main()