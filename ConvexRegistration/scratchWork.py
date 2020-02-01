from imTools import *
from scipy.spatial import ConvexHull
from convexHullPlot import plotHull, plotLowerHull
import cvxpy as cvx

def Taylor08Test(target, base):
	"""target, deformed, m x n x 3 RGB images"""
	(m,n, _ ) = target.shape

	(x,y) = (100, 100)

	errorSurface = []

	# for i in range(-5, 5):
	# 	for j in range(-5, 5):
	# 		for k in range(-5, 5):
	# 			px = np.array([i, j, k])
	# 			for a in range(-5, 5):
	# 				for b in range(-5, 5):
	# 					for c in range(-5, 5):

	# 						py = np.array([a, b, c])
	# 						# px = np.random.randint(-15, 15, 3, int)
	# 						# py = np.random.randint(-15, 15, 3, int)

	# 						(Dx, Dy) = firstOrderD((x,y), px, py)

	# 						try:
	# 							error = np.linalg.norm(target[x,y] - base[Dx,Dy], 3)
	# 							errorSurface.append([Dx, Dy, error])
	# 						except: IndexError

	for i in range(-20, 20):
		for j in range(-20, 20):

			try:
				error = np.linalg.norm(target[x,y] - base[x+i,y+j], 3)
				errorSurface.append([x+i, y+j, error])
			except: IndexError

	errorSurface = np.array(errorSurface)
	print(errorSurface)
	# # print(errorSurface[np.where(errorSurface.T[2] == errorSurface.T[2].min())[0]])
	# hull = ConvexHull(points = errorSurface)
	# avgPt = np.average(hull.points[hull.vertices], axis=0)

	# # get normals of all facets (planes) defining the convex hull:
	# normals = []

	# # Let's find all the outward normal vectors: 
	# for s in hull.simplices:
	# 	normal = np.cross(hull.points[s][1] - hull.points[s][0], hull.points[s][2] - hull.points[s][1])

	# 	normals.append(normal)

	# plotHull(errorSurface)
	plotLowerHull(errorSurface)


def firstOrderD(pix, px, py):
	"""Translation"""
	(x,y) = pix
	Dx = x + px
	Dy = y + py

	return (Dx, Dy)

def secondOrderD(pix, p):

	(x,y) = pix
	C = np.array([1, x, y, x*y, x**2, y**2])
	D = int(round(np.dot(C, p)))

	return D

def gaussianDeformation(pix, px, py):
	
	if len(px) != len(py):
		raise Exception('Invalid Deformation Parameters')

	(x,y) = pix
	k = len(px) - 3

	Dx = px[0] + x*px[1] + y*px[2] 


def deformImage(base, target, px, py):

	(m,n, _) = base.shape

	transformedImage = np.zeros([m, n, 3])
	# errorSurface = []

	for i in range(m):
		for j in range(n):
			# (Dx, Dy) = firstOrderD((i,j), px, py)
			Dx = secondOrderD((i, j), px)
			Dy = secondOrderD((i, j), py)

			try:
				transformedImage[Dx, Dy] = base[i,j]
				# errorSurface.append([i, j, np.linalg.norm(transformedImage[Dx,Dy] - target[Dx,Dy], 3)])
			except: IndexError

	
	errorSurface = []
	(x, y) = (80, 20)
	Dx = secondOrderD((x, y), px)
	Dy = secondOrderD((x, y), py)
	print(Dx, Dy)
	for i in range(-10, 10):
		for j in range(-10,10):
			try:
				errorSurface.append([i,j, np.linalg.norm(base[Dx + i , Dy + j] - target[ x , y ], 3)])
			except: IndexError

	hull = ConvexHull(points = errorSurface)

	# get normals of all facets (planes) defining the convex hull, keep only those in CH_L
	normals = []
	A = []
	B = []
	for s in hull.simplices:
		normal = np.cross(hull.points[s][1] - hull.points[s][0], hull.points[s][2] - hull.points[s][1])
		normal = normal / np.linalg.norm(normal)
		dist = np.dot(normal, hull.points[s][0])
		""" 
			For a point x to be in the lower convex hull it must be that 
			np.dot(x, normal) < dist. So the constraints should be of the form
			z(x,y) >= np.dot(normal, (dx, dy, D(x,y))) - dist 
		"""

		if normal[2] < 0: # guarantees this is for CH_L
			A.append(normal)
			B.append(dist)

	plt.imshow(transformedImage.astype(int))
	plt.show()
	# hull = ConvexHull(errorSurface)
	plotLowerHull(np.array(errorSurface))

def main():

	target = readImage('images/BrainProtonDensitySliceShifted13x17y.png')
	# target = readImage('images/BrainT1SliceR10X13Y17.png')
	base = readImage('images/BrainT1Slice.png')

	Taylor08Test(target, base)


if __name__ == '__main__':
	main()