import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull  


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

def plotLowerHull(pts):

	hull = ConvexHull(points=pts)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")
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


def main():
	pass


if __name__ == '__main__':
	main()