from imTools import *
from scipy.spatial import ConvexHull
from scipy import sparse
from convexHullPlot import plotLowerHull
from deformationFunctions import *
import cvxpy as cvx
import time


def TaylorMSConvexRegistration(target, base, deformParams, *args):
    
    if len(args) != 0:
        fullScaleFlag = args[0]
    else:
        fullScaleFlag = 1
        
    (window, numScales, bounds, deformType) = deformParams
    
    deformType = deformType.lower()
    bounds = np.array(bounds)
    
    # Set Newton Step Function:
    if len(bounds) == 0:
        NewtonStep = TaylorNSUnconstrained
    else:
        NewtonStep = TaylorNSConstrained
        
    
    P = []
    for i in range(numScales):
        
        scale = 2**((numScales - 1) - i)
        
        if fullScaleFlag == 0:
            if scale == 1:
                break
            
        dsBase = downSample(base, scale)
        dsTarget = downSample(target, scale)
        
        dsWindow = int(np.floor( window / scale ))
        
        if i == 0:
            currentBase = dsBase
        else:
            currentBase = newBase
            
        (pi, zi) = Taylor08(dsTarget, currentBase, deformType, dsWindow, scale, bounds / scale)
        
        
        P.append(-1*pi*scale)
        
        dParams = [sum(P)]

        if deformType == 'gaussian':
            sigma = 10
            dParams.append(sigma)
        if (deformType == 'gaussian') or (deformType == 'thinplate'):
            k = 4
            dParams.append(k)
        
        currentRegistration = deformImage(base, deformType, dParams)
        
        if scale != 1:
            newBase = downSample(currentRegistration, scale/2)
            
        window = int(np.floor(window*(1/2.5)))
        
        # plt.imshow(currentRegistration)
        # plt.show()
        
    p = sum(P)
    print('Current Deformation Parameters: ', p, '\n\n')

    return (p, currentRegistration)


def cvxTaylorMSConvexRegistration(target, base, deformParams, *args):
    
    if len(args) != 0:
        fullScaleFlag = args[0]
    else:
        fullScaleFlag = 1

    (window, numScales, bounds, deformType) = deformParams
    
    deformType = deformType.lower()
    bounds = np.array(bounds)
    
    if len(bounds) != 0:
        (lbX, ubX, lbY, ubY) = bounds
        
    P = []
    for i in range(numScales):
        
        scale = 2**((numScales - 1) - i)
        # dsBase = downSample(base, scale)
        dsTarget = downSample(target, scale)

        if fullScaleFlag == 0:
            if scale == 1:
                break
            
        dsWindow = int(np.floor( window / scale ))
        
        if i == 0:
            currentBase = downSample(base, scale)
        else:
            currentBase = newBase
            
        (M, N, _) = dsTarget.shape

        if deformType == 'gaussian':
            L = 19
        elif deformType == 'firstorder':
            L = 3
        elif deformType == 'secondorder':
            L = 6
        elif deformType == 'thirdorder':
            L = 10
        else:# default to gaussian
            L = 19
            deformType = 'gaussian'
        

        (Ax, Ay, Iz, b, C) = getConstraintCoeffs(dsTarget, currentBase, dsWindow, scale, deformType)
        z = cvx.Variable(M*N)
        px = cvx.Variable(L)
        py = cvx.Variable(L)

        objective = cvx.Minimize(np.ones(M*N).T @ z)
        
        constraints = [Ax @ C @ px + Ay @ C @ py - Iz @ z - b <= 0]
        if len(bounds) != 0:
            constraints = [Ax @ C @ px + Ay @ C @ py - Iz @ z - b <= 0,
                          C @ px >= lbX/scale*np.ones(M*N),
                          C @ px <= ubX/scale*np.ones(M*N),
                          C @ py >= lbY/scale*np.ones(M*N),
                          C @ py <= ubY/scale*np.ones(M*N)]
            
        prob = cvx.Problem(objective, constraints)

        result = prob.solve(solver=cvx.ECOS)
        
        pi = np.concatenate((px.value, py.value), axis=0)
        
        P.append(-1*pi*scale)
        
        dParams = [sum(P)]

        if deformType == 'gaussian':
            sigma = 10
            dParams.append(sigma)
        if (deformType == 'gaussian') or (deformType == 'thinplate'):
            k = 4
            dParams.append(k)
        
        if scale == 1:
            currentRegistration = deformImage(base, deformType, dParams)
        else:
            currentRegistration = deformImage(base, deformType, dParams)
            newBase = downSample(currentRegistration, scale/2)
        
        window = int(np.floor(window*(1/2.5)))
        
        # plt.imshow(currentRegistration)
        # plt.show()
    
    p = sum(P)
    
    return (p, currentRegistration)


def Taylor08(target, base, deformType, window, scale, dConstraints):

    deformType = deformType.lower()

    # Set Newton Step
    if len(dConstraints) == 0:
        NewtonStep = TaylorNSUnconstrained
    else:
        NewtonStep = TaylorNSConstrained

    # Get Coefficients describing convex apporixmation of MS metric
    (Ax, Ay, Iz, b, C) = getConstraintCoeffs(target, base, window, scale, deformType)

    # Set transform to 0
    if deformType == 'gaussian':
        p = np.zeros(38)
    elif deformType == 'thinplate':
        p = np.zeros(38)
    elif deformType == 'firstorder':
        p = np.zeros(6)
    elif deformType == 'secondorder':
        p = np.zeros(12)
    elif deformType == 'thirdorder':
        p = np.zeros(20)
    else:# default to second order
        p = np.zeros(12)

    L = int(len(p)/2)

    # Find a feasible z:
    z = np.ones(Iz.shape[1])

    while max(Ax @ C @ p[:L] + Ay @ C @ p[L:] -Iz @ z - b) > 0 :
        z *= 5

    t = 1
    mu = 30
    eps = 1/10**3

    M = len(b)

    while True:
        # (pStar, zStar) = TaylorNewtonStep((Ax, Ay, Iz, b, C), p, z, t)
        (pStar, zStar) = NewtonStep((Ax, Ay, Iz, b, C), p, z, t, dConstraints)
        # (pStar, zStar) = TaylorNewtonStep2((Ax, Ay, Iz, b, C), p, z, t, lbX, ubX, lbY, ubY)

        p = pStar
        z = zStar
        gap = M/t
        # print('Current gap:', gap, '\n\n')
        if gap <= eps:
            break
        t = mu*t

    return (p, z)


def getConstraintCoeffs(target, base, window, scale, deformType):
    """getConstraintCoeffs(TARGET, BASE, WINDOW, DEFORM_TYPE)
        returns (Ax, Ay, Iz, b, C) paremeters defining lower
        convex hull contraints (Ax, Ay, Iz, b) between TARGET
        wrt BASE, and deformation basis vectors C dependent on DEFORM_TYPE
    """

    deformType = deformType.lower()

    deformDictionary = {'gaussian': gaussianD, 
                        'thinplate': thinPlateD,
                        'firstorder': firstOrderD,
                        'secondorder': secondOrderD,
                        'thirdorder': thirdOrderD}

    deform = deformDictionary[deformType]

    start = time.time()
    # print("Getting Coefficients...")

    stepSize = 2
    if window <= 5:
        stepSize = 1

    (rows, cols, _) = target.shape
    # initializing coefficient matrices & vectors
    Ax = np.array([])
    Ay = np.array([])
    Iz = np.array([])
    b = []
    C = []
    numFacets = []

    dParams = scale

    if (deformType == 'gaussian') or (deformType == 'thinplate'):

        dParams = []

        kernels = getKernels((rows, cols), 4)
        dParams.append(kernels)

        if deformType == 'gaussian':
            sigma = 10
            dParams.append(sigma)

        dParams.append(scale)
    

    for y in range(rows):
        for x in range(cols):

            # Initialize Error Surface
            errorSurface = []

            """ for some predefined window, construct lower convex hull
            of error function between target[x,y] and base[x,y]
            """

            for i in range(-window, window, stepSize):
                for j in range(-window, window, stepSize):
                    if ((y + i) < rows and (y + i) >= 0 and (x+j) < cols and (x+j) >= 0): # check range
                    # try:
                        error = np.linalg.norm(target[y,x] - base[y + i, x + j], 3) ** 2
                        errorSurface.append([i, j, error])
                    # except IndexError:
                    #     pass
                    # else:
                    #   errorSurface.append([i, j, 0])
                        

            errorSurface = np.array(errorSurface)

            # Degenerate Hull Case: (typical for homogeneous region)
            if len(np.unique(errorSurface.T[2])) == 1:
            # if np.std(errorSurface.T[2]) < 10:
                Ax1 = [0]
                Ay1 = [0]
                b.append(0)

            else:
                # if (x,y) == (50, 50):
                    # plotLowerHull(errorSurface, 1)

                # hull = ConvexHull(points = errorSurface) # this has issues in homogeneous regions
                hull = ConvexHull(points = errorSurface, qhull_options='QJ') # Solves above issue
                # Initialize facet constraint lists
                Ax1 = []
                Ay1 = []
                # Get lower planar facet coefficients
                for i in range(len(hull.simplices)):
                    if hull.equations[i][2] < 0:
                        ax = hull.equations[i][1]
                        ay = hull.equations[i][0]
                        # az = hull.equations[i][2]
                        dist = -1*hull.equations[i][3]
                
                        Ax1.append(ax)
                        Ay1.append(ay)
                        # Iz1.append(1)
                        b.append(dist)

            facets = len(Ax1)
            numFacets.append(facets)
            Ax = np.append(Ax, Ax1)
            Ay = np.append(Ay, Ay1)
            Iz = np.append(Iz, np.ones(facets))

            C.append( deform((x,y), dParams) )

            # if deformType == 'gaussian':
            #     C.append(gaussianD((x,y), kernels, sigma, scale))
            # elif deformType == 'thinplate':
            #     C.append(thinPlateD((x,y), kernels, scale)) 
            # elif deformType == 'firstorder':
            #     C.append(firstOrderD((x,y), scale))
            # elif deformType == 'secondorder':
            #     C.append(secondOrderD((x,y), scale))
            # elif deformType == 'thirdorder':
            #     C.append(thirdOrderD((x,y), scale))
            # else: 
            #     C.append(secondOrderD((x,y), scale))

    # print("Done. Time elapsed:", time.time() - start, " \n\n")

    # print("Formatting Matrices...")
    
    Ax = coeffMatFormat(Ax, numFacets)
    Ay = coeffMatFormat(Ay, numFacets)
    Iz = coeffMatFormat(Iz, numFacets)
    b = np.array(b)

    C = np.array(C)

    # print("Done. Total Time Elapsed: ", time.time() - start, "\n\n")

    return (Ax, Ay, Iz, b, C)


def TaylorNSUnconstrained(facetCoeffs, p, z, t, constraints):
    # minimizes t*z - sum(log(ax_i ci.T px + ay_i ci.T py - Izi.T z - bi)) wrt z, px, py
    (Ax, Ay, Iz, b, C) = facetCoeffs

    c = t*np.ones(Iz.shape[1])
    
    # Length of px and py vectors:
    L = int(len(p) / 2)

    # line search parameters:
    alpha = 0.1
    beta = 0.8
    eps =  10**(6) 
    maxiters = 100
    count = 0
    
    # Perform Line Search:
    for iter in range(maxiters):
        
        count += 1

        (gp, gz, Hp, Hz, D6) = unconstrainedDerivatives(facetCoeffs, p[:L], p[L:], z, t)

        # Solve for Newton Step:
        g = np.concatenate((gp,gz), axis = 0)
        
        # Dinv = sparse.linalg.inv(D6)
        Dinv = sparse.diags(1/(D6.diagonal()))
        Hprime = Hp - Hz.T @ Dinv @ Hz
        gpprime = (-1 * gp) - Hz.T @ Dinv @ (-1 * gz)
        
        # Use Block Diagonals and Schur Complement to solve the system:
        # dp = np.linalg.solve(Hp - Hz.T @ Dinv @ Hz, (-1*gp) - Hz.T @ Dinv @ (-1*gz)) 
        dp = np.linalg.solve(Hprime, gpprime)
        dz = Dinv @ ((-1*gz) - Hz @ dp)
        delta = np.concatenate((dp, dz), axis = 0)
        # gprime = gp - Hz.T @ Dinv @ gz
                        
        # Check Optimality Gap:
        # lambdasqr = -1 * dp.T @ Hprime @ dp
        lambdasqr = -1 * g.T @ delta
        # print('lambdasqr/2 = ', lambdasqr/2, '\n\n')
        if lambdasqr / 2 < eps:
            break # if already eps-suboptimal
       
        # else: 
        tau = 1        
        
        # Ensure z + tau * dz is feasible
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
        
        # if count % 10 == 0:
        #   print('Newton step: ', count, '\n')
        #   print('Lambdasqr / 2 = ', lambdasqr/2, '\n\n')

    if count == maxiters:
        print('ERROR: MAXITERS reached.\n')
        p = 0
        z = 0
    
    return (p, z)


def TaylorNSConstrained(facetCoeffs, p, z, t, constraints):
    """minimizes 
    t*z - sum(log(ax_i ci.T px + ay_i ci.T py - Izi.T z - bi)) 
    - sum(log(ubX - ci.T px)) - sum(log(ci.T px - lbX)) - ... y bounds
    wrt z, px, py

    """
    (Ax, Ay, Iz, b, C) = facetCoeffs
    (lbX, ubX, lbY, ubY) = constraints

    lbX = lbX * np.ones(len(z))
    ubX = ubX * np.ones(len(z))
    lbY = lbY * np.ones(len(z))
    ubX = ubX * np.ones(len(z))

    c = t*np.ones(Iz.shape[1])
    
    # Length of px and py vectors:
    L = int(len(p) / 2)
    
    # line search parameters:
    alpha = 0.1
    beta = 0.8
    eps =  10**(6) 
    maxiters = 100
    count = 0
    
    # Perform Line Search:
    for iter in range(maxiters):
        
        count += 1
        
        (gp, gz, Hp, Hz, D6) = constrainedDerivatives(facetCoeffs, p[:L], p[L:], z, lbX, ubX, lbY, ubY, t)
        
        g = np.concatenate((gp,gz), axis = 0)
        Dinv = sparse.diags(1/(D6.diagonal()))
        
        Hprime = Hp - Hz.T @ Dinv @ Hz
        gpprime = (-1 * gp) - Hz.T @ Dinv @ (-1 * gz)
        
        # Use Block Diagonals and Schur Complement to solve the system:
        # dp = np.linalg.solve(Hp - Hz.T @ Dinv @ Hz, (-1*gp) - Hz.T @ Dinv @ (-1*gz)) 
        dp = np.linalg.solve(Hprime, gpprime)
        dz = Dinv @ ((-1*gz) - Hz @ dp)
        delta = np.concatenate((dp, dz), axis = 0)
        # gprime = gp - Hz.T @ Dinv @ gz
                        
        # Check Optimality Gap:
        #lambdasqr = -1 * dp.T @ Hprime @ dp
        lambdasqr = -1 * g.T @ delta
        # print('lambdasqr/2 = ', lambdasqr/2, '\n\n')
        if lambdasqr / 2 < eps:
            break # if already eps-suboptimal
       
        # else: 
        tau = 1        
        
        # Ensure z + tau * dz is feasible
        while ((max(Ax @ C @ (p + tau * dp)[:L] + \
                    Ay @ C @ (p + tau * dp)[L:] - \
                    Iz @ (z + tau * dz) - b) >= 0.0) \
               or (max(C @ (p + tau * dp)[:L] - ubX) >= 0.0) \
               or (max(lbX - C @ (p + tau * dp)[:L]) >= 0.0) \
               or (max(C @ (p + tau * dp)[L:] - ubY) >= 0.0) \
               or (max(lbY - C @ (p + tau * dp)[L:]) >= 0.0)):

                # Update tau
                tau = beta * tau

        # Want f(x + t*x_nt) < f(x) + t*alpha*g.T @ x_nt 
        while c.T @ (tau * dz) - sum(np.log(-1 * (Ax @ C @ (p + tau * dp)[:L] + \
                Ay @ C @ (p + tau * dp)[L:] - Iz @ (z + tau * dz) - b))) \
                - sum(np.log(ubX - C @ (p + tau * dp)[:L])) \
                - sum(np.log(C @ (p + tau * dp)[:L] - lbX)) \
                - sum(np.log(ubY - C @ (p + tau * dp)[L:])) \
                - sum(np.log(C @ (p + tau * dp)[L:] - lbY)) \
                + sum(np.log(-1 * (Ax @ C @ p[:L] + Ay @ C @ p[L:] - Iz @ z - b))) + \
                + sum(np.log(ubX - C @ p[:L])) \
                + sum(np.log(C @ p[:L] - lbX)) \
                + sum(np.log(ubY - C @ p[L:])) \
                + sum(np.log(C @ p[L:] - lbY)) \
                - alpha * tau * g.T @ delta > 0:

            # Update tau
            tau = beta * tau
                    
        p += tau * dp
        z += tau * dz
        
        # if count % 10 == 0:
        #   print('Newton step: ', count, '\n')
        #   print('Lambdasqr / 2 = ', lambdasqr/2, '\n\n')

    if count == maxiters:
        print('ERROR: MAXITERS reached.\n')
        p = 0
        z = 0
    
    return (p, z)


# FORMAT CONSTRAINT MATRICES
def coeffMatFormat(A, numFacets):
    """Converts coefficients to form in paper
    Much faster :) """
    col = np.array([])
    for i in range(len(numFacets)):
        col = np.concatenate((col, [i]*numFacets[i]), axis = 0)
        
    row = np.array([i for i in range(len(A))])
    
    return sparse.csc_matrix((A, (row, col)), shape = (len(A), len(numFacets)))


# Fullsweep of Gradients and Hessians Necessary to compute unonstrained Newton Step
def unconstrainedDerivatives(facetCoeffs, px, py, z, t):

    (Ax, Ay, Iz, b, C) = facetCoeffs
    
    # Dx = C @ px
    # Dy = C @ py
    A = (Ax @ C @ px) + (Ay @ C @ py) - Iz @ z
    # A = Ax @ Dx + Ay @ Dy - Iz @ z
    s = b - A

    d = sparse.diags(s**(-2))

    # Get Gradients:
    gz =  t - (Iz.T @ s**(-1))
    gpx =  C.T @ Ax.T @ s**(-1)
    gpy =  C.T @ Ay.T @ s**(-1)

    gp = np.concatenate((gpx, gpy), axis = 0)

    # Get Hessians:
    # Pls review Multivariate Calculus. All these signs should be good now.
    # D1 = Ax.T @ d @ Ax
    # D2 = Ax.T @ d @ Ay
    # D3 = Ay.T @ d @ Ay
    # D4 = -1 * Ax.T @ d @ Iz
    # D5 = -1 * Ay.T @ d @ Iz
    D6 = Iz.T @ d @ Iz

    # # Constructs Hp
    # Hpx   = C.T @ D1 @ C
    # Hpxpy = C.T @ D2 @ C
    # Hpy   = C.T @ D3 @ C

    Hpx = C.T @ (Ax.T @ d @ Ax) @ C
    Hpxpy = C.T @ Ax.T @ d @ Ay @ C
    Hpy = C.T @ (Ay.T @ d @ Ay) @ C

    Hp1 = np.concatenate((Hpx, Hpxpy), axis = 1)
    Hp2 = np.concatenate((Hpxpy, Hpy), axis = 1)
    Hp  = np.concatenate((Hp1, Hp2),   axis = 0)

    # Constructs Hz
    # Hz1 = D4 @ C
    # Hz2 = D5 @ C
    Hz1 = -1 * Ax.T @ d @ Iz @ C
    Hz2 = -1 * Ay.T @ d @ Iz @ C
    Hz  = np.concatenate((Hz1, Hz2), axis = 1)

    return (gp, gz, Hp, Hz, D6)

# Fullsweep of Gradients and Hessians Necessary to compute constrained Newton Step
def constrainedDerivatives(facetCoeffs, px, py, z, lbX, ubX, lbY, ubY, t):
    
    (Ax, Ay, Iz, b, C) = facetCoeffs
    
    Dx = C @ px
    Dy = C @ py
    # A = (Ax @ C @ px) + (Ay @ C @ py) - Iz @ z
    A = Ax @ Dx + Ay @ Dy - Iz @ z
    s = b - A
    sxLB = (Dx - lbX)
    sxUB = (ubX - Dx)
    syLB = (Dy - lbY)
    syUB = (ubY - Dy)
   
    dxLB = sparse.diags(sxLB**(-2))
    dxUB = sparse.diags(sxUB**(-2))
    dyLB = sparse.diags(syLB**(-2))
    dyUB = sparse.diags(syUB**(-2))

    d = sparse.diags(s**(-2))
    # d2 = d**2
    
    # Get Gradients:
    gz =  t - (Iz.T @ s**(-1))
    gpx =  C.T @ ((Ax.T @ s**(-1)) - sxLB**(-1) + sxUB**(-1))
    gpy =  C.T @ ((Ay.T @ s**(-1)) - syLB**(-1) + syUB**(-1))
    gp = np.concatenate((gpx, gpy), axis = 0)
    
    # Get Hessians:
    # Pls review Multivariate Calculus. All these signs should be good now.
    # D1 = Ax.T @ d @ Ax
    # D2 = Ax.T @ d @ Ay
    # D3 = Ay.T @ d @ Ay
    # D4 = -1 * Ax.T @ d @ Iz
    # D5 = -1 * Ay.T @ d @ Iz
    D6 = Iz.T @ d @ Iz

    # # Constructs Hp
    # Hpx   = C.T @ D1 @ C
    # Hpxpy = C.T @ D2 @ C
    # Hpy   = C.T @ D3 @ C

    Hpx = C.T @ (Ax.T @ d @ Ax + dxLB + dxUB) @ C
    Hpxpy = C.T @ Ax.T @ d @ Ay @ C
    Hpy = C.T @ (Ay.T @ d @ Ay + dyLB + dyUB) @ C

    Hp1 = np.concatenate((Hpx, Hpxpy), axis = 1)
    Hp2 = np.concatenate((Hpxpy, Hpy), axis = 1)
    Hp  = np.concatenate((Hp1, Hp2),   axis = 0)

    # Constructs Hz
    # Hz1 = D4 @ C
    # Hz2 = D5 @ C
    Hz1 = -1 * Ax.T @ d @ Iz @ C
    Hz2 = -1 * Ay.T @ d @ Iz @ C
    Hz  = np.concatenate((Hz1, Hz2), axis = 1)

    return (gp, gz, Hp, Hz, D6)

# CALCULATES HESSIAN OF BARRIER OBJECTIVE
def getHessians(Ax, Ay, Iz, C, d): # old

    # Pls review Multivariate Calculus. All these signs should be good now.
    # D1 = Ax.T @ d @ Ax
    # D2 = Ax.T @ d @ Ay
    # D3 = Ay.T @ d @ Ay
    # D4 = -1 * Ax.T @ d @ Iz
    # D5 = -1 * Ay.T @ d @ Iz
    D6 = Iz.T @ d @ Iz

    # # Constructs Hp
    # Hpx   = C.T @ D1 @ C
    # Hpxpy = C.T @ D2 @ C
    # Hpy   = C.T @ D3 @ C

    Hpx = C.T @ Ax.T @ d @ Ax @ C
    Hpxpy = C.T @ Ax.T @ d @ Ay @ C
    Hpy = C.T @ Ay.T @ d @ Ay @ C

    Hp1 = np.concatenate((Hpx, Hpxpy), axis = 1)
    Hp2 = np.concatenate((Hpxpy, Hpy), axis = 1)
    Hp  = np.concatenate((Hp1, Hp2),   axis = 0)

    # Constructs Hz
    # Hz1 = D4 @ C
    # Hz2 = D5 @ C
    Hz1 = -1 * Ax.T @ d @ Iz @ C
    Hz2 = -1 * Ay.T @ d @ Iz @ C
    Hz  = np.concatenate((Hz1, Hz2), axis = 1)

    return (Hp, Hz, D6) 


# CALCULATES GRADIENT OF BARRIER OBJECTIVE
def getGradients(Ax, Ay, Iz, C, s, t): #old
    """ Returns Gradients of barrier functions wrt p and z"""

    gz =  t - (Iz.T @ s**(-1))
    gpx =  C.T @ Ax.T @ s**(-1)
    gpy =  C.T @ Ay.T @ s**(-1)
    gp = np.concatenate((gpx, gpy), axis = 0)

    return (gp, gz)

    
def main():

    target = readImage('images/BrainT1SliceR10X13Y17.png', (50,50))
    base = readImage('images/BrainT1Slice.png', (50,50))
    (m, n, _) = base.shape

    dConstraints = [-15*np.ones(m*n), 15*np.ones(m*n), -15*np.ones(m*n), 15*np.ones(m*n)]

    (p, z) = Taylor08(target, base, 'firstOrder', 15, dConstraints)

    print(p)

    im = firstOrderDeformImage(base, p)

    plt.imshow(im)
    plt.show()


if __name__ == '__main__':
    main()