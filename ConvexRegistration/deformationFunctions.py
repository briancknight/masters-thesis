import numpy as np
from Transforms import displacementTransform

def thirdOrderD(pix, *args):

    (x,y) = pix

    if len(args)!=0:
        scale = args[0]
        x *= scale
        y *= scale
    else:
        scale = 1

    # C = np.array([1, x, y, x*y, x**2, y**2])
    return np.array([1, x, y, x*y, x**2, y**2, x**3, x**2 * y, x * y**2, y**3])

def secondOrderD(pix, *args):
    
    (x,y) = pix

    if len(args)!=0:
        scale = args[0]
        x *= scale
        y *= scale
    else:
        scale = 1

    # C = np.array([1, x, y, x*y, x**2, y**2])
    return np.array([1, x, y, x*y, x**2, y**2])

def firstOrderD(pix, *args):

    (x,y) = pix

    if len(args)!=0:
        scale = args[0]
        x *= scale
        y *= scale
    else:
        scale = 1

    return np.array([1, x, y])

def gaussianD(pix, *args):

    (x,y) = pix

    if len(args[0])!=3:
        (kernels, sigma) = args[0]
        scale = 1
    elif len(args[0]) == 3:
        (kernels, sigma, scale) = args[0]

    x *= scale
    y *= scale


    C = [1, x, y]

    for kernel in kernels:
        r = np.linalg.norm(np.array([x, y]) -  np.array(kernel) * scale, 2)
        C.append(np.exp(-r/(sigma**2)))

    # return int(round(np.dot(C, p)))
    return np.array(C)

def thinPlateD(pix, *args):

    (x,y) = pix

    if len(args[0])!=2:
        kernels = args[0][0]
        scale = 1
    elif len(args[0]) == 2:
        (kernels, scale) = args[0]
    
    x *= scale
    y *= scale
    C = [1, x, y]

    for kernel in kernels:
        r = np.linalg.norm(np.array([x, y]) -  np.array(kernel)*scale, 2)
        C.append((r**2)*np.log(r))

    # return int(round(np.dot(C, p)))
    return np.array(C)

def getKernels(dims, k):
    """
    im should be a numpy array, this returns evenly
    distributed seed points for segmentation algorithms

    returns k^2 seeds, evently distributed in im
    """
    (rows, cols) = dims

    if k**2 > max(rows, cols):
        raise Exception('Image too small for this many kernels')

    kernels = []

    if k == 1:
        kernel = (int(np.floor(rows / 2)), int(np.floor(cols / 2)))
        kernels.append(kernel)

        return kernels

    else:
        xStart = np.floor(cols / k)
        xEnd = cols - xStart

        yStart = np.floor(rows / k)
        yEnd = rows - yStart

        x = np.linspace(xStart, xEnd, k).astype(int)
        y = np.linspace(yStart, yEnd, k).astype(int)

        grid, _ = np.meshgrid(x, y)

        for i in range(k):
            for j in range(k):

                kernels.append((grid[0][i], grid[0][j]))

        return kernels


def thirdOrderDeformImage(im, params, *args):

    if len(args)!=0:
        scale = args[0]
    else:
        scale = 1

    p = params[0]

    if len(p) != 20:
        raise Exception('Incorrect number of deformation parameters for Third Order Deformation')

    px = p[:10]
    py = p[10:]

    (rows, cols, _) = im.shape

    displacements = np.zeros([rows, cols, 2])

    for y in range(rows):
        for x in range(cols):

            c = thirdOrderD((x, y), scale)
            Dx = np.dot(c, px)
            Dy = np.dot(c, py)
            # The minuse defines our orientation (down and right is  ++)
            displacements[y, x] = [-Dx, -Dy]

    return displacementTransform(im, displacements)


def secondOrderDeformImage(im, params, *args):

    if len(args)!=0:
        scale = args[0]
    else:
        scale = 1

    p = params[0]

    if len(p) != 12:
        raise Exception('Incorrect number of deformation parameters for Second Order Deformation')

    px = p[:6]
    py = p[6:]

    (rows, cols, _) = im.shape

    displacements = np.zeros([rows, cols, 2])

    for y in range(rows):
        for x in range(cols):

            c = secondOrderD((x, y), scale)
            Dx = np.dot(c, px)
            Dy = np.dot(c, py)
            # The minuse defines our orientation (down and right is  ++)
            displacements[y, x] = [-Dx, -Dy]

    return displacementTransform(im, displacements)


def firstOrderDeformImage(im, params, *args):

    if len(args)!=0:
        scale = args[0]
    else:
        scale = 1

    # Models an affine transformation
    p = params[0]

    if len(p) != 6:
        raise Exception('Incorrect number of deformation parameters for First Order Deformation')

    px = p[:3]
    py = p[3:]

    (rows, cols, _) = im.shape

    displacements = np.zeros([rows, cols, 2])

    for y in range(rows):
        for x in range(cols):

            c = firstOrderD((x,y))
            Dx = np.dot(c, px)
            Dy = np.dot(c, py)
            displacements[y, x] = [-Dx, -Dy]

    return displacementTransform(im, displacements)

    
def gaussianDeformImage(im, params, *args):
    
    if len(args)!=0:
        scale = args[0]
    else:
        scale = 1

    (p, sigma, k) = params

    if len(p) != 38:
        raise Exception('Incorrect number of deformation parameters for Gaussian Deformation')

    px = p[:19]
    py = p[19:]

    (rows,cols, _) = im.shape
    
    kernels = getKernels((rows, cols), k)

    displacements = np.zeros([rows, cols, 2])
    
    for y in range(rows):
        for x in range(cols):
    
            c = gaussianD((x,y), [kernels, sigma, scale])
            Dx = np.dot(c, px)
            Dy = np.dot(c, py)
            displacements[y, x] = [-Dx, -Dy]

    return displacementTransform(im, displacements)

def thinPlateDeformImage(im, params, *args):
   
    if len(args)!=0:
        scale = args[0]
    else:
        scale = 1

    (p, k) = params

    if len(p) != 38:
        raise Exception('Incorrect number of deformation parameters for Second Order Deformation')

    px = p[:19]
    py = p[19:]

    (rows,cols, _) = im.shape

    kernels = getKernels((rows, cols), k)

    displacements = np.zeros([rows, cols, 2])

    for y in range(rows):
        for x in range(cols):

            c = thinPlateD((x,y), [kernels,scale])
            Dx = np.dot(c, px)
            Dy = np.dot(c, py)
            displacements[y, x] = [-Dx, -Dy]

    return displacementTransform(im, displacements)


def deformImage(im, deformType, dParams, *args):

    if len(args)!=0:
        scale = args[0]
    else:
        scale = 1

    deformDictionary = {'gaussian': gaussianDeformImage, 
                        'thinplate': thinPlateDeformImage,
                        'firstorder': firstOrderDeformImage,
                        'secondorder': secondOrderDeformImage,
                        'thirdorder': thirdOrderDeformImage}

    deformation = deformDictionary[deformType.lower()]

    return deformation(im, dParams, scale)