import cvxpy as cvx
import numpy as np


def leastsquares(A, b, *args):

	# Takes in optional upper / lower bounds
	if A.shape[0] != b.shape[0]:

		raise Exception('Invalid Dimensions')

	(m,n) = A.shape
	x = cvx.Variable(n)
	constraints = []

	if len(args) !=0:
		if len(args[0]) == 2:
			l = args[0][0]
			u = args[0][1]

			constraints = [u - x >= np.zeros(n), x - l >= np.zeros(n)]

		if len(args[0]) == 1:
			l = args[0][0]

			constraints = [x - l >= np.zeros(n)]

	objective = cvx.Minimize(cvx.atoms.norm(A*x - b))
	prob = cvx.Problem(objective, constraints)
	result = prob.solve()

	return (x.value, result)

def main():

	# m = 100
	# n = 55

	# A = np.random.rand(m,n)
	# b = np.random.rand(m)

	# u = np.random.rand(n)
	# l = u - 20*np.ones(n)

	# (x, result) = leastsquares(A,b, [l,u])

	# print(x, result)


	A = np.random.randint(-100, 100, [50, 100])
	c = np.random.randint(-100, 100, 100)
	b = np.random.randint(-100, 100, 50)
	
	(x, p) = leastsquares(A, b, [np.zeros(len(c))])


if __name__ == '__main__':
	main()