import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from imTools import *
import scipy.signal


def firstDerivs(pixArray):
	"""Will take a numpy array of size (x, y, 3) and return an array of first derivatives
 This is an array of shape (x, y, 6, 3)
 For each pixel, the first elements of (4, 3) array are as follow:
 minus-x, plus-x, mid-x, minus-y, plus-y, mid-y, where mid-x and mid-y are averages of the minus and plus derivatives respectively
 """

	(xDim, yDim, z) = pixArray.shape

	xOffMinus = np.append(pixArray[:, 0, :].reshape(xDim, 1, z), pixArray[:, :-1, :], axis=1)
	xOffPlus = np.append(pixArray[:, 1:, :], pixArray[:, -1, :].reshape(xDim, 1, z), axis=1)

	yOffMinus = np.append(pixArray[0, :, :].reshape(1, yDim, z), pixArray[:-1, :, :], axis=0)
	yOffPlus = np.append(pixArray[1:, :, :], pixArray[-1, :, :].reshape(1, yDim, z), axis=0)

	xMinus = pixArray - xOffMinus
	xPlus = xOffPlus - pixArray
	xMid = .5 * (xMinus + xPlus)

	yMinus = pixArray - yOffMinus
	yPlus = yOffPlus - pixArray
	yMid = .5 * (yMinus + yPlus)

	xMinus = np.expand_dims(xMinus, axis=2)
	xPlus = np.expand_dims(xPlus, axis=2)
	xMid = np.expand_dims(xMid, axis=2)

	yMinus = np.expand_dims(yMinus, axis=2)
	yPlus = np.expand_dims(yPlus, axis=2)
	yMid = np.expand_dims(yMid, axis=2)

	derivs = np.concatenate((xMinus, xPlus, xMid, yMinus, yPlus, yMid), axis=2)

	return derivs



def updateImage(fixLambda, derivMatrix, h, origImage, curImage):
	"""
	Takes lambda parameter, a derivative matrix given by firstDerivs, 
	parameter h, the original image and the current image, 
	and returns the Bounded Variation component of the decomposition given by Rudin Osher and Fatemi (ROF TV Paper citation here)
	"""
	xOffMinus = np.pad(curImage[:, :-1, :], [[0, 0], [1, 0], [0, 0]], 'constant')
	xOffPlus = np.pad(curImage[:, 1:, :], [[0, 0], [0, 1], [0, 0]], 'constant')
	yOffMinus = np.pad(curImage[:-1, :, :], [[1, 0], [0, 0], [0, 0]], 'constant')
	yOffPlus = np.pad(curImage[1:, :, :], [[0, 1], [0, 0], [0, 0]], 'constant')

	cE = ceFunc(curImage, derivMatrix, .01)
	cW = cwFunc(curImage, derivMatrix, .01)
	cS = csFunc(curImage, derivMatrix, .01)
	cN = cnFunc(curImage, derivMatrix, .01)

	denom = (2 * fixLambda * (h ** 2)) + cE + cW + cS + cN

	num = ((2 * fixLambda * (h ** 2) * origImage) + (cE * xOffPlus) + (cW * xOffMinus) +
		(cS * yOffPlus) + (cN * yOffMinus))

	return np.array(num / denom)



def ceFunc(pixArray, derivs, epsilon):
	""" 
	takes pixel array, derivative matrix, and epsilon factor for computational purposes.
	returns c_e matrix as defined in ROF TV Paper
	"""
	DxPlus = derivs[:, :, 1, :]
	DyMid = derivs[:, :, 5, :]

	denominator = np.sqrt( ( epsilon ** 2 ) + ( DxPlus ** 2 ) + ( DyMid ** 2 )  )

	ce = ( 1.0 / denominator ) 
	return ce



def cwFunc(pixArray, derivs, epsilon):
	""" 
	takes pixel array, derivative matrix, and epsilon factor for for computing..
	returns c_w matrix as defined in ROF TV Paper
	"""	
	xOffMinus = np.pad(pixArray[:, :-1, :], [[0, 0], [1, 0], [0, 0]], 'constant')
	xOffMinusDerivs = firstDerivs(xOffMinus)

	DxMinus = derivs[:, :, 0, :]
	xOffDyMid = xOffMinusDerivs[:, :, 5, :]

	denominator = np.sqrt( ( epsilon ** 2 ) + ( DxMinus ** 2 ) + ( xOffDyMid ** 2 )  )
	
	cw = ( 1.0 / denominator )
	return cw



def csFunc(pixArray, derivs, epsilon):
	""" 
	takes pixel array, derivative matrix, and epsilon factor for computing..
	returns c_s matrix as defined in ROF TV Paper
	"""
	DxMid = derivs[:, :, 2, :]
	DyPlus = derivs[:, :, 4, :]

	denominator = np.sqrt( ( epsilon ** 2 ) + ( DxMid ** 2) + ( DyPlus ** 2) )

	cs = ( 1.0 / denominator )
	return cs



def cnFunc(pixArray, derivs, epsilon):
	""" 
	takes pixel array, derivative matrix, and epsilon factor for computing.
	returns c_n matrix as defined in ROF TV Paper
	"""
	yOffMinus = np.pad(pixArray[:-1, :, :], [[1, 0], [0, 0], [0, 0]], 'constant')
	yOffMinusDerivs = firstDerivs(yOffMinus)

	yOffDxMid = yOffMinusDerivs[:, :, 2, :]
	DyMinus = derivs[:, :, 3, :]

	denominator = np.sqrt( ( epsilon ** 2 ) + ( yOffDxMid ** 2) + ( DyMinus ** 2) )
	cn = ( 1.0 / denominator )
	return cn



def iterateUpdates(fixLambda, iterations, curIteration, origImage, curImage):
	"""
	Takes in starting lambda, number of iterations, 
	the current iteration number, the original image, and the current image.
	Runs ROF algorithm iteratively for specifed number of iterations.
	This is written recursively.
	"""
	if curIteration == iterations:
		return curImage

	derivMatrix = firstDerivs(curImage)
	h = 1.0 / curImage.size
	newImage = updateImage(fixLambda, derivMatrix, h, origImage, curImage)

	return iterateUpdates(fixLambda, iterations, curIteration+1, origImage, newImage)



# This runs the multi-scale algorithm and returns lists of the lambdas
# and the nu's.

def minimizeFunctional(fixLambda, iterations, curIteration, origImage, uImageSeq, vImageSeq):
	"""
	Takes in initial lambda, number of iterations (number of scales), the current iteration,
	an original image, and sequence of u_lambda's and v_lambda's as defined
	in the TV ROF paper. It returns both the u_lambda sequence and the v_lambda sequence,
	which correspond to BV components and L^2 components resepctively. 
	This is written recursively.
	"""
	if curIteration == iterations:
		print('Multi-Scale decomposition complete.')
		return (uImageSeq, vImageSeq)

	if curIteration == 0:
		print('Iterating: ', curIteration)
	else: 
		print('...')

	newImage = iterateUpdates(fixLambda, 10, 1, origImage, np.array(origImage))
	newOrig = origImage - newImage
	return minimizeFunctional(2*fixLambda, iterations, curIteration+1,
		newOrig, uImageSeq+[newImage], vImageSeq+[newOrig])



def testLambdas():

	fixLambda = 0.0


	while fixLambda <= .01:
		count = 0
		print('looping')
		im = readImage('../images/barbara.png')
		fig = plt.figure()
		a = fig.add_subplot(3, 3, 1)
		plt.axis('off')
		plt.imshow(im.astype(int))
		(uImageSeq, vImageSeq) = minimizeFunctional(fixLambda, 7, 0, im, [], [])

		for i in range(len(uImageSeq)):
			
			curIm = np.zeros(im.shape)

			for j in range(i+1):

				curIm += uImageSeq[j]

			count += 1
			a = fig.add_subplot(3, 3, i+2)
			plt.axis('off')
			plt.imshow(curIm.astype(int))

		plt.savefig('outImages/lambda='+str(fixLambda)+'.png')


		fixLambda += .0005



def meanDenoise(im):

	derivMatrix = firstDerivs(im)
	xMid = derivMatrix[:, :, 2, :]
	yMid = derivMatrix[:, :, 5, :]
	avgMid = .5 * (xMid + yMid)

	return np.array(im + avgMid).astype(int)

def medianFilter(pixArray, size):
	"""medianFilter(pixArray, size): applies a size x size median filter to the RGB image array and returns the filtered array. 
	
	For edge values we pad pixArray accordingly and apply the filter to relevant pixels."""
	# print(pixArray.shape)

	# ***This is slow, don't use it. Alternatively: medianIm = scipy.signa.medfilt(pixArray, [size, size, 1])***
	(m, n, rgb) = pixArray.shape

	paddedPixArray = np.pad(pixArray, [[size,size],[size,size],[0,0]], 'median')

	medianIm = np.zeros([m,n,rgb])

	for i in range(size, m+size):
		for j in range(size, n+size):
			medianIm[i-size,j-size,0] = np.median(paddedPixArray[(i-size):(i+size),(j-size):(j+size),0])
			medianIm[i-size,j-size,1] = np.median(paddedPixArray[(i-size):(i+size),(j-size):(j+size),1])
			medianIm[i-size,j-size,2] = np.median(paddedPixArray[(i-size):(i+size),(j-size):(j+size),2])

	# print(medianIm.shape)
	return medianIm



def runROFAlgorithm(imagePath, size):

	im = readImage(imagePath, size)
	saltPepperIm = saltPepperNoise(im, 2)

	newIm = iterateUpdates(.0005, 10, 1, saltPepperIm, saltPepperIm)

	plt.figure()
	plt.imshow(newIm.astype(int))
	plt.figure()
	plt.imshow(saltPepperIm)
	plt.figure()
	plt.imshow(im)
	plt.show()

	return
