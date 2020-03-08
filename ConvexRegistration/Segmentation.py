import numpy as np
import SimpleITK as sitk
from MultiScaleRep import *
from Transforms import *
import scipy.misc


def binarySegmentation(im, lower, upper, inside, outside):

	def binSegComponent(im, lower, upper, inside, outside, rgb):

		im = sitk.GetImageFromArray(im, sitk.sitkVectorFloat32)
		im = sitk.VectorIndexSelectionCast(im, rgb, sitk.sitkFloat32)
		segIm = sitk.BinaryThreshold(im, lower, upper, inside, outside)
		
		return sitk.GetArrayFromImage(segIm).astype(int)

	r = binSegComponent(im, lower, upper, inside, outside, 0)
	g = binSegComponent(im, lower, upper, inside, outside, 1)
	b = binSegComponent(im, lower, upper, inside, outside, 2)

	return pieceRGB(r, g, b)



def OtsuSegmentation(im, inside, outside):

	def OtsuSegComponent(im, inside, outside, rgb):

		im = sitk.GetImageFromArray(im, sitk.sitkVectorFloat32)
		im = sitk.VectorIndexSelectionCast(im, rgb, sitk.sitkFloat32)
		otsuFilter = sitk.OtsuThresholdImageFilter()
		otsuFilter.SetInsideValue(inside)
		otsuFilter.SetOutsideValue(outside)
		segIm = otsuFilter.Execute(im)
		return sitk.GetArrayFromImage(segIm).astype(int)

	if im.shape[2] != 3:
		return OtsuSegmentationComponent(im, inside, outside, 0)

	r = OtsuSegComponent(im, inside, outside, 0)
	g = OtsuSegComponent(im, inside, outside, 1)
	b = OtsuSegComponent(im, inside, outside, 2)

	return pieceRGB(r, g, b)



def connectedThreshold(im, lower, upper, replaceVal, *args):

	def conThreshComponent(im, lower, upper, replaceVal, rgb):

		image = sitk.GetImageFromArray(im, sitk.sitkVectorFloat32)
		image = sitk.VectorIndexSelectionCast(image, rgb, sitk.sitkFloat32)
		#
		# Blur using CurvatureFlowImageFilter
		#
		blurFilter = sitk.CurvatureFlowImageFilter()
		blurFilter.SetNumberOfIterations(5)
		blurFilter.SetTimeStep(0.125)
		image = blurFilter.Execute(image)

		#
		# Set up ConnectedThresholdImageFilter for segmentation
		#
		segmentationFilter = sitk.ConnectedThresholdImageFilter()
		segmentationFilter.SetLower(float(lower))
		segmentationFilter.SetUpper(float(upper))
		segmentationFilter.SetReplaceValue(replaceVal)

		for seed in seeds:
			(seedX, seedY) = seed
			seed = [int(seedX), int(seedY)]
			segmentationFilter.AddSeed(seed)
		# print("Adding seed at: ", seed, " with intensity: ", image.GetPixel(*seed))

		# Run the segmentation filter
		image = segmentationFilter.Execute(image)
		# image[seed] = 255

		return sitk.GetArrayFromImage(image).astype(int)
	print(args)
	if len(args) > 0:
		print(args[0])
		nSeeds = args[0]
	else:
		nSeeds = 1

	print(nSeeds)
	seeds = getSeeds(im, nSeeds)
	print(seeds)

	r = conThreshComponent(im, lower, upper, 255, 0)
	g = conThreshComponent(im, lower, upper, 255, 1)
	b = conThreshComponent(im, lower, upper, 255, 2)

	return pieceRGB(r, g, b)



def fastMarching(im, params, *args):
	"""
	Takes in image to be segmented, (x, y) initialization
	 seed, sigma, alpha, beta, parameters, not sure what
	 these do yet, and time threshold and stopping time.

	(10, 1.0, -0.5,  3.0, 1000, 1000) seem to work well with ../images/coins.png

	Returns binary segmentation filter (image)

	*args is a image filter object, defaults to binary thresholding filter 
	with no input arg
	"""
	(nSeeds, sigma, alpha, beta, tThresh, T) = params

	# get seeds
	seeds = getSeeds(im, nSeeds)

	im = sitk.GetImageFromArray(im, sitk.sitkVectorFloat32)

	(r, g, b) = getSITKComponents(im)
		
	def fastMarchingComp(im):

		smoothing = sitk.CurvatureAnisotropicDiffusionImageFilter()
		smoothing.SetTimeStep(0.125)
		smoothing.SetNumberOfIterations(5)
		smoothing.SetConductanceParameter(9.0)
		smoothingOutput = smoothing.Execute(im)

		gradientMagnitude = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
		gradientMagnitude.SetSigma(sigma)
		gradientMagnitudeOutput = gradientMagnitude.Execute(smoothingOutput)

		sigmoid = sitk.SigmoidImageFilter()
		sigmoid.SetOutputMinimum(0.0)
		sigmoid.SetOutputMaximum(1.0)
		sigmoid.SetAlpha(alpha)
		sigmoid.SetBeta(beta)
		# sigmoid.DebugOn()
		sigmoidOutput = sigmoid.Execute(gradientMagnitudeOutput)


		fastMarching = sitk.FastMarchingImageFilter()

		i = 0

		for seed in seeds:
			seedValue = i
			(seedX, seedY) = seed

			trialPoint = (int(seedX), int(seedY), seedValue)
			fastMarching.AddTrialPoint(trialPoint)
			i += 1

		fastMarching.SetStoppingValue(T)

		fastMarchingOutput = fastMarching.Execute(sigmoidOutput)

		if len(args) > 0:
			thresholder = args[0]
		else:
			thresholder = sitk.BinaryThresholdImageFilter()
			thresholder.SetLowerThreshold(0.0)
			thresholder.SetUpperThreshold(tThresh)
			thresholder.SetOutsideValue(0)
			thresholder.SetInsideValue(255)

		result = thresholder.Execute(fastMarchingOutput)

		# sitk.WriteImage(result, outputFilename)

		return sitk.GetArrayFromImage(result).astype(int)

	r = fastMarchingComp(r)
	g = fastMarchingComp(g)
	b = fastMarchingComp(b)

	return pieceRGB(r, g, b)



def pieceRGB(r, g, b):

	r = np.expand_dims(r, axis=2)
	g = np.expand_dims(g, axis=2)
	b = np.expand_dims(b, axis=2)

	return np.concatenate((r, g, b), axis=2)


def toSeq(im1, im2):

	tVals = .01 * np.array(range(0, 100, 2))
	return [((1-t)*im2+t*im1) for t in tVals]

def getSITKComponents(sitkIm):

	red = sitk.VectorIndexSelectionCast(sitkIm, 0, sitk.sitkFloat32)
	green = sitk.VectorIndexSelectionCast(sitkIm, 1, sitk.sitkFloat32)
	blue = sitk.VectorIndexSelectionCast(sitkIm, 2, sitk.sitkFloat32)

	return (red, green, blue)


def getSeeds(im, nSeeds):
	"""
	im should be a numpy array, this returns evenly
	distributed seed points for segmentation algorithms

	returns n^2 seeds, evently distributed in im
	"""
	(m, n, rgb) = im.shape

	if nSeeds > max(m, n):
		raise Exception('Image too large for this many seeds')

	seeds = []

	if nSeeds == 1:
		seed = (int(np.floor(m / 2)), int(np.floor(n / 2)))
		seeds.append(seed)

		return seeds

	else:
		xStart = np.floor(m / nSeeds)
		xEnd = m - xStart

		yStart = np.floor(n / nSeeds)
		yEnd = n - yStart

		x = np.linspace(xStart, xEnd, nSeeds).astype(int)
		y = np.linspace(yStart, yEnd, nSeeds).astype(int)

		grid, _ = np.meshgrid(x, y)

		for i in range(nSeeds):
			for j in range(nSeeds):

				seeds.append((grid[0][i], grid[0][j]))

		return seeds


def overlayImages(im, overlay, *args):

	if len(args) > 0:
		opacity = args[0]
	else:
		opacity = 0.5

	plt.figure()
	plt.imshow(im.astype(int))
	plt.imshow(overlay.astype(int), alpha = opacity)

	return


def multiScaleCoarseRegistration(origIm, txIm, method, metricName, lambdaTerm):
	# origIm is original image, txIm is tranformed image

	# Decompose original and transformed images:
	(origUSeq, origVseq) = minimizeFunctional(
						lambdaTerm, 10, 0, origIm, [], [])

	(txUSeq, txVSeq) = minimizeFunctional(
						lambdaTerm, 10, 0, txIm, [], [])

	# Register two coarse scales:
	(registeredCoarse, tx) = registerImages(
		origUSeq[0], txUSeq[0], method, metricName)

	registeredIm = transformImage(origIm, tx)

	return (registeredIm.astype(int), tx)


def multiScaleCoarseSegmentation(origIm, txIm, method, metricName, lambdaTerm):

	(_, tx) = multiScaleCoarseRegistration(
				origIm, txIm, method, metricName, lambdaTerm)
	# Segment Original Image:
	origSeg = fastMarching(origIm, 
				(10, 1.0, -0.5,  3.0, 50000, 50000))

	return (transformImage(origSeg, tx).astype(int), tx)


def evaluateCoarseSeg(origIm, noisyTx, cleanTx, method, lambdaTerm):

	registeredSegmentation = multiScaleCoarseSegmentation(
								origIm, noisyTx, method, 0.0005)

	actualSeg = fastMarching(cleanTx, 
				(10, 1.0, -0.5,  3.0, 50000, 50000))

	(MS, NC, MI) = evaluateSegmentation(
						registeredSegmentation, actualSeg)


	return (MS, NC, MI)