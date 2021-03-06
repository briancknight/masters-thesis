import cvxpy as cvx
import cvxopt
import numpy as np

def optimalActivity(A, cmax, p, pdisc, q):
	# See Problem 4.17 Boyd Text

	(m,n) = A.shape
	x = cvx.Variable(n)
	rev = cvx.sum( cvx.minimum( p*x, p*q + pdisc*(x - q) ) ) 
	objective = cvx.Maximize( rev )
	constraints = [x >= np.zeros(n), cvx.matmul(A,x) <= cmax]

	prob = cvx.Problem(objective, constraints)
	prob.solve(solver=cvx.CPLEX)

	return (x.value)


def optimalActivity2(A, cmax, p, pdisc, q):
	# See Problem 4.17 Boyd Text

	(m,n) = A.shape
	u = cvx.Variable(n)
	x = cvx.Variable(n)
	objective = cvx.Maximize( cvx.sum( u ) )
	constraints = [x >= np.zeros(n), cvx.matmul(A,x) <= cmax, cvx.matmul(p, x) >=u, cvx.matmul(p, q) + cvx.matmul(pdisc, x-q) >= u]

	prob = cvx.Problem(objective, constraints)
	result = prob.solve()

	totrev = sum(np.minimum(p*x.value, p*q + pdisc*(x.value - q)))

	return (x.value, totrev)


def main():

	# A = np.array([[1, 2, 0, 1],
	# 			  [0, 0, 3, 1], 
	# 			  [0, 3, 1, 1], 
	# 			  [2, 1, 2, 5], 
	# 			  [1, 0, 3, 2]])
	A = np.array([[1, 2, 0, 1],
				  [0, 0, 3, 1], 
				  [0, 3, 1, 1], 
				  [2, 1, 2, 5], 
				  [1, 0, 3, 2]])

	cmax = np.array([100, 100, 100, 100, 100])
	p = np.array([3, 2, 7, 6])
	pdisc = np.array([2, 1, 4, 2])
	q = np.array([4, 10, 5, 10])

	x = optimalActivity(A, cmax, p, pdisc, q)

	print('x = ', x)
	r = np.minimum(p*x, p*q + pdisc*(x-q))
	totr = sum(r)
	print(r)
	print(totr)
	# print(x, p)
	# print('x* = ')
	# print(x)
	# print('\n\n')
	# print('p* = ')
	# print(p)


if __name__ == '__main__':
	main()