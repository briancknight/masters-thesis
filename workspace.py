import numpy as np
from PIL import Image
from astar import astar
# this function will define a 2-dimensional work space and it's obstacles.

def main():

	m = 500 # rows
	n = 700 # cols

	obs1 = makeObstacle(m, n, intervalToIndices([[100,150], [200,300]]))
	obs2 = makeObstacle(m, n, intervalToIndices([[20,80], [450,520]]))
	obs3 = makeObstacle(m, n, intervalToIndices([[300,380], [40,120]]))

	obs = [obs1, obs2, obs3]

	W = workspace(m, n, obs)

	# img = Image.fromarray(W)
	# img.show()

	path = astar(W.tolist(), (350,25), (0, 499))
	# path = np.array(path)

	path = makeObstacle(m, n, path)

	obs = [obs1, obs2, obs3, path]

	W = workspace(m, n, obs)

	img = Image.fromarray(W)
	img.show()


def workspace(m, n, Obs):
	"""
	m, n should be integers defining the size of the workspace, Obs should be an array of matrices of obstacles.
	"""
	W = np.zeros((m,n))

	for Ob in Obs:
		W += Ob

	return W

def makeObstacle(m, n, indices):

	obs = np.zeros((m,n))

	for index in indices:
		obs[index[0], index[1]] = 255

	return obs

def intervalToIndices(I):
	# Takes in interval in R^2 and converts set of indices
	# I = [[a,b], [c,d]]

	indices = []

	for i in range(I[0][0], I[0][1]):
		for j in range(I[1][0],I[1][1]):
			indices.append([i,j])

	return indices

def CES(W, Pi, Pf, q, qd, qdd, otherstuff):
	pass

def getBubbles(path, obs, rl, ru):
	"""Generates Bubbles around waypoints in path"""

	for p in path[1:-1]:

		if 
def GenerateBubble(Pi):
	# retuns bubble for point Pi
	pass

def TranslateBubble(Ai, ri):
	""" returns bubble with translated center"""
	pass

if __name__ == '__main__':
	main()