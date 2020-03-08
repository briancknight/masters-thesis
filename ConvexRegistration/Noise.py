import numpy as np
from MultiScaleRep import *
from Transforms import *


def saltPepperNoise(pixArray, percent, *args ):
	# saltPepperNoise(pixArray, percent, *args) takes np.array of RGB image, noise percentage,
	# *args: light intesity value, probability, dark intensity value, probability, returns noisy image
	if len(args) != 0:
		(liteInt, litePr, darkInt, darkPr) = args
	else:
		# set parameters for standard salt and pepper noise:
		liteInt = 255
		darkInt = 0
		litePr = 0.5
		darkPr = 0.5

	(m, n, rgb) = pixArray.shape

	# Get probabilities and distribution
	percent = percent * 0.01
	litePr = litePr * percent
	darkPr = darkPr * percent
	probability = [1 - (litePr + darkPr), litePr, darkPr]

	# Determine which pixels to corrupt
	corrupt = np.random.choice([np.NaN, liteInt, darkInt], (m,n), p=probability)
	saltLists = np.where(corrupt == liteInt)
	saltCoords = list( zip(saltLists[0], saltLists[1]) )
	pepperLists = np.where(corrupt == darkInt)
	pepperCoords = list( zip(pepperLists[0], pepperLists[1]) )

	# Corrupt image
	spIm = np.copy(pixArray)
	for i in saltCoords:
		spIm[i] = np.array([liteInt,liteInt,liteInt])
	for i in pepperCoords:
		spIm[i] = np.array([darkInt, darkInt, darkInt])

	return spIm



# def gaussianNoise(im, amount):

# 	if len(im.shape) != 3:
# 		raise Exception('Not a color image')


# 	newIm = np.array(im, dtype=float)
# 	noise = np.random.normal(0, amount, im.shape)
# 	newIm += noise
# 	return newIm.astype(int)

def gaussianNoise(pixArray, mu, sigma, *args):
	# takes np.array of image, desired mean and standard deviation of gaussian noise,
	# returns noisy image
	(m, n, d) = pixArray.shape

	if len(args) > 0:
		grayFlag = args[0]
	else:
		grayFlag = 0

	if grayFlag == 0:
		noise = np.random.normal(mu, sigma, (m, n, d))
		noise = np.array(noise, dtype=float)

	else:
		noiseComp = np.random.normal(mu, sigma, (m, n))
		noise = pieceRGB(noiseComp, noiseComp, noiseComp)

	noisyIm = pixArray.astype(np.uint8) + noise

	return noisyIm.astype(np.uint8)

def gaussianNoiseMult(im, mu, sigma, *args):

	if len(im.shape) != 3:
		raise Exception('Not a color image')

	if len(args) > 0:
		grayFlag = args[0]
	else:
		grayFlag = 0

	(m, n, rgb) = im.shape

	newIm = np.array(im, dtype=np.uint8)

	if grayFlag == 0:
		noise = np.random.normal(mu, sigma, (m, n, d))
		noise = np.array(noise, dtype=np.uint8)

	else:
		noiseComp = np.random.normal(mu, sigma, (m, n))
		noise = pieceRGB(noiseComp, noiseComp, noiseComp)

	newIm = newIm.astype(np.uint8) * noise

	return newIm.astype(np.uint8)