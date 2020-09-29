import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull 
from imTools import *


def plotHull(pts):

	hull = ConvexHull(points=pts)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")
	# Plot defining corner points
	ax.plot(hull.points.T[0], hull.points.T[1], hull.points.T[2], "ko")	

	# Find vertices of boundary points & Plot them:
	hullPts = []
	for s in hull.simplices:
		hullPts.append(hull.points[s][0])
		hullPts.append(hull.points[s][1])
		hullPts.append(hull.points[s][2])

	hullPts = np.array(hullPts)
	ax.plot(hullPts.T[0], hullPts.T[1], hullPts.T[2], "bo")

	# Plot Facets:
	for i in range(len(hull.simplices)):
		s = hull.simplices[i]
		# Append the first point again to complete triangle plot:
		s = np.append(s, s[0])
		ax.plot(hull.points[s, 0], hull.points[s, 1], hull.points[s, 2], 'g-')

	# Make axis label
	for i in ["x", "y", "z"]:
	    eval("ax.set_{:s}label('{:s}')".format(i, i))


	plt.show()

def plotLowerHull(pts, scatterFlag):

	# hull = ConvexHull(points=pts)
	hull = ConvexHull(points = pts, qhull_options='QJ')

	fig = plt.figure(figsize=(6.5,6.5))
	ax = fig.add_subplot(111, projection="3d")

	if scatterFlag == 1:
		ax.scatter(pts.T[0], pts.T[1], pts.T[2], '.')
	# Plot defining corner points
	# ax.plot(hull.points.T[0], hull.points.T[1], hull.points.T[2], "ko")	

	# Plot Facets with negative z-component of outward normal vector, collect constraint coefficients:
	A = []
	B = []
	constraints = []
	lowerFacets = []

	for i in range(len(hull.simplices)):
		lowerFacets = np.append(lowerFacets, i)
		s = hull.simplices[i]
		if hull.equations[i][2] < 0:
			a = hull.equations[i][:3]
			b = hull.equations[i][3]
			s = np.append(s, s[0])


			ax.plot(hull.points[s].T[0], hull.points[s].T[1], hull.points[s].T[2], "bo")
			ax.plot(hull.points[s, 0], hull.points[s, 1], hull.points[s, 2], 'g-')
			# Defining a_i's and b_i's for each lower facet (s_i):

	# Make axis label
	for i in ["x", "y", "z"]:
	    eval("ax.set_{:s}label('{:s}')".format(i, i))

	plt.show()

def plotl2Scatter(pts):

	fig = plt.figure(figsize=(6.5,6.5))
	ax = fig.add_subplot(111, projection="3d")
	pts = np.array(pts)
	ax.scatter(pts.T[0], pts.T[1], pts.T[2], '.r-')

	# Make axis label
	for i in ["x", "y", "z"]:
	    eval("ax.set_{:s}label('{:s}')".format(i, i))

	plt.show()


def generateExampleFigures():

	base = readImage('images/BrainT1Slice.png')
	target = readImage('images/BrainT1SliceR10X13Y17.png')

	# Example of feature region
	(x, y) = (125, 110)

	pts = []
	pts2 = np.zeros((20, 20))
	for i in range(-12, 12):
		for j in range(-12, 12):

			pts.append([i, j, np.linalg.norm(target[y,x] - base[y+i,x+j], 3)**2])
			pts2[i,j] = np.linalg.norm(target[y,x] - base[y+i,x+j], 3)**2
	pts = np.array(pts)

	plotl2Scatter(pts)
	plotLowerHull(pts, 1)
	plotLowerHull(pts, 0)
	plt.imshow(pts2, cmap = 'hot')
	plt.show()
	# plt.colorbar()

	# Example of homogeneous region
	(x, y) = (150, 75)

	pts = []
	pts2 = np.zeros((20, 20))

	for i in range(-12, 12):
		for j in range(-12, 12):

			pts.append([i, j, np.linalg.norm(target[y,x] - base[y+i,x+j], 3)**2])
			pts2[i,j] = np.linalg.norm(target[y,x] - base[y+i,x+j], 3)**2
	pts = np.array(pts)

	plotl2Scatter(pts)
	plotLowerHull(pts, 1)
	plotLowerHull(pts, 0)
	plt.imshow(pts2, cmap = 'hot')
	plt.show()

def main():
	
	# generateExampleFigures()
	ptCloud = np.random.rand(10, 2)
	hull = ConvexHull(ptCloud)

	fig = plt.figure()

	plt.plot(ptCloud[:,0], ptCloud[:,1], 'o')

	plt.show()

	plt.plot(ptCloud[:,0], ptCloud[:,1], 'o')

	for i in range(len(hull.simplices)):
			s = hull.simplices[i]
			if hull.equations[i][1] < 0:
				a = hull.equations[i][:2]
				b = hull.equations[i][2]
				s = np.append(s, s[0])


				plt.plot(hull.points[s].T[0], hull.points[s].T[1], "bo")
				plt.plot(hull.points[s, 0], hull.points[s, 1], 'g-')

	plt.show()

	plt.plot(ptCloud[:,0], ptCloud[:,1], 'o')
	
	for i in range(len(hull.simplices)):
			s = hull.simplices[i]
			# if hull.equations[i][1] < 0:
			a = hull.equations[i][:2]
			b = hull.equations[i][2]
			s = np.append(s, s[0])


			plt.plot(hull.points[s].T[0], hull.points[s].T[1], "bo")
			plt.plot(hull.points[s, 0], hull.points[s, 1], 'r-')

	plt.show()
if __name__ == '__main__':
	main()