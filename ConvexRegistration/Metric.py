import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from Transforms import *
import time


# README:  All metrics work on R, G, and B individually, so they
# will return a 1 x 3 matrix with the metric applied to each of the
# components.


# Calculates Mean Squares metric, see paper

def meanSquaresMetric(A, B):

	if A.shape != B.shape:
		raise Exception('Images are not the same shape')
	A = A / 255.0
	B = B / 255.0
	(m, n, rbg) = A.shape
	diffSquare = np.power(A-B, 2)
	return np.sum(diffSquare, axis=(0, 1)) / (m*n)

def meanSquaresMetric3D(A, B):

	# this actually returns the sum of the output from meanSquaresMetric,
	# which is interesting (maybe?) but probably not useful
	if A.shape != B.shape:
		raise Exception('Images are not the same shape')

	size = A[:,:,0].size # = m x n

	A = A / 255.0
	B = B / 255.0

	squaredDiffIm = np.power(A-B, 2)

	sumSquared = np.sum(np.sum(squaredDiffIm, axis=2))

	return sumSquared / size


# Calculates normal correlation metric.  Images are the same if they
# return -1.  WARNING:  These numbers get BIG and will overflow the
# allowable space Python provides ints if you take the color values to be
# 0 - 255.  You can work around this with numpy.array.astype(uint64)

def normCorrMetric(A, B):

	if A.shape != B.shape:
		raise Exception('Images are not the same shape')

	A = A / 255.0
	B = B / 255.0
	product = np.sum(A*B, axis=(0, 1))
	summedA = np.sum(A*A, axis=(0, 1))
	summedB = np.sum(B*B, axis=(0, 1))
	denomSq = (summedA * summedB)
	return -1.0 * product / (np.sqrt(denomSq + .0001))


# Returns 3 histograms for one image.  RGB

def getHistograms(im):

	if len(im.shape) != 3:
		raise Exception('Not a color image')

	(m, n, rgb) = im.shape
	flatIm = np.reshape(im, (m*n, rgb))
	bins = np.array(range(257))

	(histR, _) = np.histogram(flatIm[:, 0], bins=bins)
	(histG, _) = np.histogram(flatIm[:, 1], bins=bins)
	(histB, _) = np.histogram(flatIm[:, 2], bins=bins)

	return (histR, histG, histB, bins)


# Return 3 normalized histograms for one image.  RGB

def getNormalHistograms(im):

	(histR, histG, histB, bins) = getHistograms(im)
	(m, n, _) = im.shape
	pixelNum = float(m * n)

	normHistR = histR / pixelNum
	normHistG = histG / pixelNum
	normHistB = histB / pixelNum

	return (normHistR, normHistG, normHistB, bins)


# Returns a R, G, and B histograms over two images

def getJointHistograms(im1, im2):

	if im1.shape != im2.shape:
		raise Exception('Images are not the same shape')

	if len(im1.shape) != 3:
		raise Exception('Images are not rgb')

	(m, n, rgb) = im1.shape
	flatIm1 = np.reshape(im1, (m*n, rgb))
	flatIm2 = np.reshape(im2, (m*n, rgb))
	bins = (np.array(range(257)), np.array(range(257)))

	(histR, _, _) = np.histogram2d(flatIm1[:, 0], flatIm2[:, 0], bins) 
	(histG, _, _) = np.histogram2d(flatIm1[:, 1], flatIm2[:, 1], bins)
	(histB, _, _) = np.histogram2d(flatIm1[:, 2], flatIm2[:, 2], bins)

	return (histR, histG, histB, bins)


# Returns normalized histograms for two images

def getNormalJointHistograms(im1, im2):

	(histR, histG, histB, bins) = getJointHistograms(im1, im2)
	(m, n, _) = im1.shape
	pixelNum = m * n

	return (histR/pixelNum, histG/pixelNum, histB/pixelNum, bins)



# Will calculate a single xlog(x) term for an input.
# Notice the zero check

def singleEntropy(x):

	if abs(x) < .001:
		return 0.0

	if x < 0.0:
		raise Exception('Negative values encountered in entropy')

	return x * np.log2(x)


# Fun function.  Will take in either the getNormalHistogram functions
# or the normalHistogram function and apply 

def getImageEntropy(im):

	vecEntropy = np.vectorize(singleEntropy)
	(histR, histG, histB, _) = getNormalHistograms(im)

	(m, n, _) = im.shape

	red = histR[im[:, :, 0]]
	green = histG[im[:, :, 1]]
	blue = histB[im[:, :, 2]]

	coeff = -1.0 / (m * n)

	redEntropy = np.sum(vecEntropy(red), axis=(0, 1))
	greenEntropy = np.sum(vecEntropy(green), axis=(0, 1))
	blueEntropy = np.sum(vecEntropy(blue), axis=(0, 1))

	return (coeff*redEntropy, coeff*greenEntropy, coeff*blueEntropy)


# Like the getEntropy function except that it will calculate the 
# joint entropy of two images.  Notice how we index into the joint
# histogram.  Don't muck this one up

def getJointEntropy(im1, im2):

	vecEntropy = np.vectorize(singleEntropy)
	(histR, histG, histB, _) = getNormalJointHistograms(im1, im2)

	(m, n, _) = im1.shape

	red = histR[im1[:, :, 0], im2[:, :, 0]]
	green = histG[im1[:, :, 1], im2[:, :, 1]]
	blue = histB[im1[:, :, 2], im2[:, :, 2]]

	coeff = -1.0 / (m * n)

	redEntropy = np.sum(vecEntropy(red), axis=(0, 1))
	greenEntropy = np.sum(vecEntropy(green), axis=(0, 1))
	blueEntropy = np.sum(vecEntropy(blue), axis=(0, 1))

	return (coeff*redEntropy, coeff*greenEntropy, coeff*blueEntropy)



# Will calculate the mutualInformation through the entropy

def calculateMutualInfo(im1, im2):

	A = im1
	B = im2
	(r1, g1, b1) = getImageEntropy(A)
	(r2, g2, b2) = getImageEntropy(B)

	(rBoth, gBoth, bBoth) = getJointEntropy(A, B)

	r = (r1 + r2) - rBoth
	g = (g1 + g2) - gBoth
	b = (b1 + b2) - bBoth

	return (r, g, b)


def paperMutualInfo(im1, im2):
	
	(rJoint, gJoint, bJoint, _) = getNormalJointHistograms(im1, im2)
	(r1, g1, b1, _) = getNormalHistograms(im1)
	(r2, g2, b2, _) = getNormalHistograms(im2)

	pJoint = np.stack((rJoint, gJoint, bJoint), axis = 2)

	r12 = np.outer(r1, r2)
	g12 = np.outer(g1, g2)
	b12 = np.outer(b1, b2)
	pProd = np.stack((r12, g12, b12), axis = 2)

	nzIndSep = pProd.nonzero()
	nzIndicesBottom = list( zip(nzIndSep[0], nzIndSep[1], nzIndSep[2]) )

	product = np.zeros(pProd.shape)
	for index in nzIndicesBottom:
		if pJoint[index] != 0: #leaves 0*log(0) = 0
			product[index] = pJoint[index] * np.log2(pJoint[index] / pProd[index])

	mutualInfo = np.sum(product, axis = (0,1))

	return mutualInfo