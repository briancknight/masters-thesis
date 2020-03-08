from Transforms import *
from Metric import *
from Segmentation import *
import SimpleITK as sitk


def getSITKComponents(sitkIm):

	red = sitk.VectorIndexSelectionCast(sitkIm, 0, sitk.sitkFloat32)
	green = sitk.VectorIndexSelectionCast(sitkIm, 1, sitk.sitkFloat32)
	blue = sitk.VectorIndexSelectionCast(sitkIm, 2, sitk.sitkFloat32)

	return (red, green, blue)


def commandIteration(method):

	print('{0:3} = {1:10.5f} : {2}'.format(method.GetOptimizerIteration(),
		method.GetMetricValue(),
		method.GetOptimizerPosition()))


def registerImages(im1, im2, method, metricName, *args):

	moving = sitk.GetImageFromArray(im1, sitk.sitkVectorFloat32)
	fixed = sitk.GetImageFromArray(im2, sitk.sitkVectorFloat32)

	if len(args) > 0:
		initializeTx = args[0]
		reg = initializeRegistration(moving, fixed, 
						method, metricName, initializeTx)
	else:
		reg = initializeRegistration(moving, fixed, 
									method, metricName)	

	(tx, regVal) = getBestTx(moving, fixed, reg)

	registeredIm = transformImage(im1, tx)

	return (registeredIm.astype(int), tx)


def multiNodalRegistration(im1, im2, method, metric, lambdaTerm, m, *args):

	# set number of scales to register, defaults to all
	cutoff = m
	if len(args) > 0:
		cutoff = args[0]

	# decompose images:
	(uIm1Seq, vIm1Seq) = minimizeFunctional(lambdaTerm, m, 0, im1, [], [])
	(uIm2Seq, vIm2Seq) = minimizeFunctional(lambdaTerm, m, 0, im2, [], [])

	# register by increasing scale:
	for i in range(cutoff):

		if i == 0:
			(registration, initialTx) = registerImages(
					uIm1Seq[i], uIm2Seq[i], method, metric)

			currTx = initialTx

		else:
			initialTx = currTx

			(registration, tx) = registerImages(
				sum(uIm1Seq[:i+1]), sum(uIm2Seq[:i+1]), 
				method, metric, currTx)
			currTx = tx

	return (registration, tx)

def multiNodalRegistration2(uIm1Seq, uIm2Seq, method, metric, lambdaTerm, *args):

	if len(uIm1Seq) != len(uIm2Seq):
		raise Exception('Image Sequences are not of the same length')

	m = len(uIm1Seq)
	# set number of scales to register, defaults to all
	cutoff = m
	if len(args) > 0:
		cutoff = args[0]

	# register by increasing scale:
	for i in range(cutoff):

		if i == 0:
			(registration, initialTx) = registerImages(
					uIm1Seq[i], uIm2Seq[i], method, metric)

			currTx = initialTx

		else:
			initialTx = currTx

			(registration, tx) = registerImages(
				sum(uIm1Seq[:i+1]), sum(uIm2Seq[:i+1]), 
				method, metric, currTx)
			currTx = tx

	return (registration, tx)

def initializeRegistration(moving, fixed, method, metricName, *args):

	if len(args) > 0:
		initTx = args[0]

		if method.lower() == 'translation':
			reg = initializeTranslation(metricName, fixed, moving, initTx)
		elif method.lower() == 'bspline':
			reg = initializeBSpline(metricName, fixed, moving, initTx)
		elif method.lower() == 'euler':
			reg = initializeEuler2D(metricName, fixed, moving, initTx)
		elif method.lower() == 'similarity':
			reg = initializeSimilarity2D(metricName, fixed, moving, initTx)
	else:
		if method.lower() == 'translation':
			reg = initializeTranslation(metricName, fixed, moving)
		elif method.lower() == 'bspline':
			reg = initializeBSpline(metricName, fixed, moving)
		elif method.lower() == 'euler':
			reg = initializeEuler2D(metricName, fixed, moving)
		elif method.lower() == 'similarity':
			reg = initializeSimilarity2D(metricName, fixed, moving)

	return reg


def getBestTx(moving, fixed, registration):

	(rTx, rVal) = registerComponent(moving, fixed, registration, 0)
	(gTx, gVal) = registerComponent(moving, fixed, registration, 1)
	(bTx, bVal) = registerComponent(moving, fixed, registration, 2)

	rgbVals = abs(np.array([rVal, gVal, bVal]))

	rgbTxs = (rTx, gTx, bTx)

	best = np.where(rgbVals == min(rgbVals))[0][0]

	return (rgbTxs[best], best)


def registerComponent(moving, fixed, registration, rgb):

	movingComp = sitk.VectorIndexSelectionCast(moving, rgb, sitk.sitkFloat32)
	fixedComp = sitk.VectorIndexSelectionCast(fixed, rgb, sitk.sitkFloat32)

	tx = registration.Execute(fixedComp, movingComp)

	return (tx, registration.GetMetricValue())
	

def initializeTranslation(metricName, fixed, moving, *args):

	reg = sitk.ImageRegistrationMethod()

	if metricName.lower() == 'ms':
		reg.SetMetricAsMeanSquares()

	elif metricName.lower() == 'nc':
		reg.SetMetricAsCorrelation()

	elif metricName.lower() == 'mi':
		reg.SetMetricAsJointHistogramMutualInformation()

	else:
		reg.SetMetricAsMeanSquares()

	reg.SetMetricSamplingPercentage(.75)
	reg.SetMetricSamplingStrategy(reg.RANDOM)
	reg.SetOptimizerAsRegularStepGradientDescent(1.0, .00001, 500)

	if len(args) > 0:
		initialTransform = args[0]
	else:
		initialTransform = sitk.TranslationTransform(fixed.GetDimension())

	reg.SetInitialTransform(initialTransform)
	reg.SetInterpolator(sitk.sitkLinear)

	return reg

def initializeBSpline(metricName, fixed, moving, *args):

	movingComp = sitk.VectorIndexSelectionCast(moving, 0, sitk.sitkFloat32)
	fixedComp = sitk.VectorIndexSelectionCast(fixed, 0, sitk.sitkFloat32)
	reg = sitk.ImageRegistrationMethod()

	if metricName.lower() == 'ms':
		reg.SetMetricAsMeanSquares()

	elif metricName.lower() == 'nc':
		reg.SetMetricAsCorrelation()

	elif metricName.lower() == 'mi':
		reg.SetMetricAsJointHistogramMutualInformation()

	elif metricName.lower() == 'mattes':
		reg.SetMetricAsMattesMutualInformation(50)

	else:
		reg.SetMetricAsMeanSquares()

	transformDomainMeshSize = [8]*moving.GetDimension()

	if len(args) > 0:
		initialTransform = args[0]
	else:
		initialTransform = sitk.BSplineTransformInitializer(
			fixed, transformDomainMeshSize)

	reg.SetOptimizerAsGradientDescentLineSearch(5.0, 100,
                                          convergenceMinimumValue=1e-4,
                                          convergenceWindowSize=5)
	reg.SetOptimizerScalesFromPhysicalShift( )
	reg.SetInterpolator(sitk.sitkLinear)
	reg.SetInitialTransform(initialTransform, True)

	return reg

def initializeEuler2D(metricName, fixed, moving, *args):

	movingComp = sitk.VectorIndexSelectionCast(moving, 0, sitk.sitkFloat32)
	fixedComp = sitk.VectorIndexSelectionCast(fixed, 0, sitk.sitkFloat32)
	reg = sitk.ImageRegistrationMethod()

	if metricName.lower() == 'ms':
		reg.SetMetricAsMeanSquares()

	elif metricName.lower() == 'nc':
		reg.SetMetricAsCorrelation()

	elif metricName.lower() == 'mi':
		reg.SetMetricAsJointHistogramMutualInformation()

	elif metricName.lower() == 'mattes':
		reg.SetMetricAsMattesMutualInformation(50)

	else:
		reg.SetMetricAsMattesMutualInformation(50)

	reg.SetMetricSamplingPercentage(.75)
	reg.SetMetricSamplingStrategy(reg.RANDOM)
	reg.SetOptimizerAsRegularStepGradientDescent(1.0, .0001, 500)
	
	if len(args) > 0:
		initialTransform = args[0]
	else:
		initialTransform = sitk.CenteredTransformInitializer(fixedComp, 
	                                      movingComp, 
	                                      sitk.Euler2DTransform(), 
	                                      sitk.CenteredTransformInitializerFilter.MOMENTS)

	reg.SetInitialTransform(sitk.Transform(initialTransform))
	reg.SetInterpolator(sitk.sitkLinear)

	return reg

def initializeSimilarity2D(metricName, fixed, moving, *args):

	movingComp = sitk.VectorIndexSelectionCast(moving, 0, sitk.sitkFloat32)
	fixedComp = sitk.VectorIndexSelectionCast(fixed, 0, sitk.sitkFloat32)
	reg = sitk.ImageRegistrationMethod()

	if metricName.lower() == 'ms':
		reg.SetMetricAsMeanSquares()

	elif metricName.lower() == 'nc':
		reg.SetMetricAsCorrelation()

	elif metricName.lower() == 'mi':
		reg.SetMetricAsJointHistogramMutualInformation()

	elif metricName.lower() == 'mattes':
		reg.SetMetricAsMattesMutualInformation(50)

	else:
		reg.SetMetricAsMeanSquares()

	reg.SetOptimizerAsRegularStepGradientDescent(learningRate=0.5,
                                           minStep=1e-4,
                                           numberOfIterations=500,
                                           gradientMagnitudeTolerance=1e-8 )
	reg.SetOptimizerScalesFromIndexShift()

	if len(args) > 0:
		initialTransform = args[0]
	else:
		initialTransform = sitk.CenteredTransformInitializer(
		fixedComp,
	 	movingComp, 
	 	sitk.Similarity2DTransform(),
	 	sitk.CenteredTransformInitializerFilter.MOMENTS)

	reg.SetInitialTransform(initialTransform)

	reg.SetInterpolator(sitk.sitkLinear)

	return reg