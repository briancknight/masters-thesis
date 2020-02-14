import numpy as np
np.warnings.filterwarnings('ignore')
rcond = -1
from leastsquares import leastsquares
from matplotlib import pyplot as plt
from lp_solver import LPSolver
import cvxpy as cvx
"""This will e a simple implementation of the barrier method for a general LP"""

def eqNewtonStep(A, b, c, x):

	# parameters for line search:
	alpha = 0.01
	beta = 0.8
	eps = 1/(10**6)
	maxiters = 100

	if (max(x) <= 0) or (np.linalg.norm(A@x - b) > 1/(10^3)):
		print(max(x) <= 0)
		print(np.linalg.norm(A @ x - b) > 1/(10^3))
		print('FAILED (infeasible starting point)')

		return (0, 0, [])

	m = len(b)
	n = len(x)

	lambda_hist = []
	count = 0

	for iter in range(maxiters):
	    count += 1
	    H = np.diag(1/x**2)
	    g = c - (1/x)

	    w = np.linalg.solve(A @ np.diag(x**2) @ A.T, -1*A @ np.diag(x**2) @ g)
	    dx = -1 * np.diag(x**2) @ (A.T @ w + g)

	    lambdasqr = -1*g.T @ dx
	    lambda_hist.append(lambdasqr/2)

	    if lambdasqr/2 <= eps:
	        break

	    # otherwise perform line search:
	    t = 1
	    while min(x + t*dx) <= 0:
	        t = beta * t
	        
	    while c.T * t @ dx - np.sum(np.log(x + t * dx)) + np.sum(np.log(x)) - alpha * t * (g.T @ dx) > 0:
	        t = beta * t

	    x = x + t * dx

	if count == maxiters:
		print('ERROR: MAXITERS reached.\n')
		x = 0
		w = 0 
	return (x, w, lambda_hist)

def ineqNewtonStep(A, b, c, x):

	# parameters for line search:
	alpha = 0.01
	beta = 0.8
	eps = 1/(10**6)
	maxiters = 100

	if min(b - A @ x) <= 0:
		print('FAILED (infeasible starting point)')
		return (0, [])

	m = len(b)
	n = len(x)

	lambda_hist = []
	count = 0
	d = b - A @ x

	for iter in range(maxiters):
	    count += 1
	    H = A.T @ np.diag(1/(d**2)) @ A
	    g = c + A.T @ (1/d)

	    # w = np.linalg.solve(A @ np.diag(x**2) @ A.T, -1*A @ np.diag(x**2) @ g)
	    # dx = -1 * np.diag(x**2) @ (A.T @ w + g)
	    dx = -1 * np.linalg.inv(H) @ g 
	    print(dx)

	    lambdasqr = -1*g.T @ dx
	    lambda_hist.append(lambdasqr/2)

	    if lambdasqr/2 <= eps:
	        break

	    # otherwise perform line search:
	    t = 1
	    while min(b - A @ (x + t*dx)) <= 0:
	        t = beta * t
	    
	    while ( c.T * t @ dx ) - ( A.T @ np.sum(np.log(b - A @ (x + t * dx))) ) +\
	    ( A.T @ np.sum(np.log(b - A @ (x + t * dx))) @ A ) -\
	    ( alpha * t * (g.T @ dx) ) > 0:

	        t = beta * t

	    x = x + t * dx

	if count == maxiters:
		print('ERROR: MAXITERS reached.\n')
		x = 0

	return (x, lambda_hist)


def lpEqBarrier(A, b, c, x):
	"""
	Solves the equality constrained minimization problem
	minimize c^Tx
	subject to Ax = b

	Given a strictly feasible starting point x"""

	(m, n) = A.shape

	t = 1
	mu = 20
	eps = 1/(10**3)

	history = []

	while True:
		(xStar, v, lambda_hist) = eqNewtonStep(A, b, t*c, x)
		x = xStar
		gap = n / t
		history.append([lambda_hist, gap])

		if gap <= eps:
			break
		t = mu*t

	return (xStar, history, gap)


def lpIneqBarrier(A, b, c, x):
	"""
	Solves the inequality constrained minimization problem
	minimize c^Tx
	subject to Ax <= b

	Given a strictly feasible starting point x

	Does not currently work :(
	"""

	(m, n) = A.shape

	t = 1
	mu = 20
	eps = 1/(10**3)

	history = []

	while True:
		(xStar, lambda_hist) = ineqNewtonStep(A, b, t*c, x)
		x = xStar
		gap = m / t
		history.append([lambda_hist, gap])

		if gap <= eps:
			break
		t = mu*t

	return (xStar, history, gap)


def plotGapVSteps(history):

	plt.figure()

	count = 0
	for hist in history:
		plt.plot(range(count, len(hist[0]) + count), hist[1]*np.ones(len(hist[0])))
		count+= len(hist[0]) - 1

	plt.xlabel('Newton Step')
	plt.ylabel('Duality Gap')
	plt.show()

def plotLambdaConvergence(lambda_hist):

	plt.figure()

	plt.scatter(range(len(lambda_hist)), lambda_hist)
	plt.xlabel('Newton Step')
	plt.ylabel('lambda^2/2')

def main():

	# Creates a random but feasible problem
	(m, n) = (100, 500)
	A = np.random.randn(m,n)
	A[0,:] = np.random.rand(n) + 0.1
	p = np.random.rand(n) + 0.1
	b = A @ p
	b = b + 10
	c = np.random.randn(n)

	# get feasible x:
	# x0, p = leastsquares(A, b, [np.zeros(len(c))])

	# (xopt, v, lambda_hist) = NewtonStep(A, b, c, x0)

	# (xStar, history, gap) = lpEqBarrier(A, b, c, x0)
	(xStar, history, gap) = lpIneqBarrier(A, b, c, p)
	print(c.T @ xStar)

	y = cvx.Variable(n)
	objective = cvx.Minimize(c.T @ y)
	constraints = [A @ y <= b]
	problem = cvx.Problem(objective, constraints)
	result = problem.solve()

	print(result)



if __name__ == '__main__':
	main()

