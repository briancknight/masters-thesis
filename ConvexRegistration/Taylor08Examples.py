from Taylor08 import *
from Metrcs import meanSquaresMetric3D

def DeformedBrainPDExample():

	basePath = 'images/BrainProtonDensitySlice.png'
	targetPath = 'images/DeformedBrainPD.png'

	base = readImage(basePath)
	target = readImage(targetPath)

	start = time.time()

	# deformParams = (30, 3, [-50, 50, -50, 50], 'gaussian')
	# 	deformParams = (27, 3, [-50, 50, -50, 50], 'gaussian') current registration parameters
	deformParams = (35, 3,[], 'gaussian')
	(p, rIm) = TaylorMSConvexRegistration(target, base, deformParams, 1)

	print('Total Time elapsed: ', (time.time() - start)/60, ' minutes\n\n')

	msDiff = meanSquaresMetric3D(target, rIm)

	print('Deformation Coefficients: ', p, '\n\n')
	print('Registration Parameters: ', deformParams, '\n\n')
	print('Mean Squares Difference: ', msDiff, '\n\n')

	pilIm = Image.fromarray(np.uint8(rIm))
	# pilIm.save('/Users/brknight/Documents/ConvexOptimization/figures/DeformedBrainPDEx.png')
	pilIm.save('/Users/brknight/Documents/ConvexOptimization/figures/DeformedBrainPDUnboundedExTest.png')


def WarpedBrainPDExample():

	basePath = 'images/BrainProtonDensitySlice.png'
	targetPath = 'images/WarpedBrainProtonDensitySlice.png'

	base = readImage(basePath)
	target = readImage(targetPath)

	start = time.time()

	deformParams = (40, 3, [-50, 50, -50, 50], 'gaussian')
	(p, rIm) = cvxTaylorMSConvexRegistration(target, base, deformParams, 1)

	print('Total Time elapsed: ', (time.time() - start)/60, ' minutes\n\n')

	msDiff = meanSquaresMetric3D(target, rIm)

	print('Deformation Coefficients: ', p, '\n\n')
	print('Registration Parameters: ', deformParams, '\n\n')

	pilIm = Image.fromarray(np.uint8(rIm))
	pilIm.save('/Users/brknight/Documents/ConvexOptimization/figures/WarpedBrainPDEx.png')


def BrainMidSaggitalGaussianExample():

	basePath = 'images/BrainMidSagittalSlice.png'
	targetPath = 'images/DefMid.jpg'

	base = readImage(basePath, (187, 155))
	target = readImage(targetPath)

	start = time.time()

	#deformParams = (20, 3, [-50, 50, -50, 50], 'gaussian')
	deformParams = (20, 3, [], 'gaussian')

	(p, rIm) = TaylorMSConvexRegistration(target, base, deformParams, 1)

	print('Total Time elapsed: ', (time.time() - start)/60, ' minutes\n\n')

	msDiff = meanSquaresMetric3D(target, rIm)

	print('Deformation Coefficients: ', p, '\n\n')
	print('Registration Parameters: ', deformParams, '\n\n')
	print('Mean Squares Difference: ', msDiff, '\n\n')

	pilIm = Image.fromarray(np.uint8(rIm))
	pilIm.save('/Users/brknight/Documents/ConvexOptimization/figures/BrainMidSaggitalGaussianUnboundedEx.png')

def webGLBrainMidSaggitalGaussianExample():

	basePath = 'images/webGLMid.png'
	targetPath = 'images/webGLwarpedMid.png'

	base = readImage(basePath)
	target = readImage(targetPath)

	print(base.shape)

	start = time.time()

	#deformParams = (20, 3, [-50, 50, -50, 50], 'gaussian')
	deformParams = (30, 3, [], 'gaussian')

	(p, rIm) = TaylorMSConvexRegistration(target, base, deformParams, 1)

	print('Total Time elapsed: ', (time.time() - start)/60, ' minutes\n\n')

	msDiff = meanSquaresMetric3D(target, rIm)

	print('Deformation Coefficients: ', p, '\n\n')
	print('Registration Parameters: ', deformParams, '\n\n')
	print('Mean Squares Difference: ', msDiff, '\n\n')

	pilIm = Image.fromarray(np.uint8(rIm))
	pilIm.save('/Users/brknight/Documents/ConvexOptimization/figures/webGLMidEx.png')

def BrainMidSaggitalThirdOrderExample():

	basePath = 'images/BrainMidSagittalSlice.png'
	targetPath = 'images/DefMid.jpg'

	base = readImage(basePath, (187, 155))
	target = readImage(targetPath)

	start = time.time()

	deformParams = (30, 3, [-50, 50, -50, 50], 'thirdOrder')
	(p, rIm) = cvxTaylorMSConvexRegistration(target, base, deformParams, 1)

	print('Total Time elapsed: ', (time.time() - start)/60, ' minutes\n\n')

	msDiff = meanSquaresMetric3D(target, rIm)

	print('Deformation Coefficients: ', p, '\n\n')
	print('Registration Parameters: ', deformParams, '\n\n')
	print('Mean Squares Difference: ', msDiff, '\n\n')

	pilIm = Image.fromarray(np.uint8(rIm))
	pilIm.save('/Users/brknight/Documents/ConvexOptimization/figures/BrainMidSaggitalThirdOrderEx.png')


def BrainT1ShiftedRotatedEx():

	basePath = 'images/BrainT1Slice.png'
	targetPath = 'images/BrainT1SliceR10X13Y17.png'
	base = readImage(basePath)
	target = readImage(targetPath)

	start = time.time()

	# deformParams = (35, 3, [], 'firstOrder')
	deformParams = (50, 3, [], 'firstOrder')

	(p, rIm) = TaylorMSConvexRegistration(target, base, deformParams, 1)
	# (p, rIm) = cvxTaylorMSConvexRegistration(target, base, deformParams, 1)

	print('Total Time elapsed: ', (time.time() - start)/60, ' minutes\n\n')

	msDiff = meanSquaresMetric3D(target, rIm)

	print('Deformation Coefficients: ', p, '\n\n')
	print('Registration Parameters: ', deformParams, '\n\n')
	print('Mean Squares Difference: ', msDiff, '\n\n')

	pilIm = Image.fromarray(np.uint8(rIm))
	pilIm.save('/Users/brknight/Documents/ConvexOptimization/figures/Braint1ShiftedRotatedEx50.png')


def BrainPDShiftedX13Y17Example():

	basePath = 'images/BrainProtonDensitySlice.png'
	targetPath = 'images/BrainProtonDensitySliceShifted13x17y.png'

	base = readImage(basePath)
	target = readImage(targetPath)

	start = time.time()

	deformParams = (30, 3, [], 'firstOrder')
	(p, rIm) = TaylorMSConvexRegistration(target, base, deformParams, 1)

	print('Total Time elapsed: ', (time.time() - start)/60, ' minutes\n\n')

	msDiff = meanSquaresMetric3D(target, rIm)

	print('Deformation Coefficients: ', p, '\n\n')
	print('Registration Parameters: ', deformParams, '\n\n')
	print('Mean Squares Difference: ', msDiff, '\n\n')

	pilIm = Image.fromarray(np.uint8(rIm))
	pilIm.save('/Users/brknight/Documents/ConvexOptimization/figures/BrainPDShiftedX13Y17Ex2.png')


def WarpedRiceExample():

	basePath = 'images/rice.png'
	targetPath = 'images/WarpedRice.png'

	base = readImage(basePath, (300,300))
	target = readImage(targetPath)

	start = time.time()

	deformParams = (50, 3, [-50, 50, -50, 50], 'gaussian') # 1
	deformParams = (30, 3, [-50, 50, -50, 50], 'gaussian') # 2
	(p, rIm) = TaylorMSConvexRegistration(target, base, deformParams, 1)

	print('Total Time elapsed: ', (time.time() - start)/60, ' minutes\n\n')

	msDiff = meanSquaresMetric3D(target, rIm)

	print('Deformation Coefficients: ', p, '\n\n')
	print('Registration Parameters: ', deformParams, '\n\n')
	print('Mean Squares Difference: ', msDiff, '\n\n')

	pilIm = Image.fromarray(np.uint8(rIm))
	pilIm.save('/Users/brknight/Documents/ConvexOptimization/figures/WarpedRiceEx2.png')


def WarpedRiceCVXExample():

	basePath = 'images/rice.png'
	targetPath = 'images/WarpedRice.png'

	base = readImage(basePath, (300,300))
	target = readImage(targetPath)

	start = time.time()

	deformParams = (50, 3, [-50, 50, -50, 50], 'gaussian')
	(p, rIm) = cvxTaylorMSConvexRegistration(target, base, deformParams, 1)

	print('Total Time elapsed: ', (time.time() - start)/60, ' minutes\n\n')

	msDiff = meanSquaresMetric3D(target, rIm)

	print('Deformation Coefficients: ', p, '\n\n')
	print('Registration Parameters: ', deformParams, '\n\n')
	print('Mean Squares Difference: ', msDiff, '\n\n')

	pilIm = Image.fromarray(np.uint8(rIm))
	pilIm.save('/Users/brknight/Documents/ConvexOptimization/figures/WarpedRiceCVXEx.png')


def MiddleburryExample():
	basePath = 'images/barn1/im0.ppm'
	targetPath = 'images/barn1/im8.ppm'

	base = readImage(basePath)
	target = readImage(targetPath)

	start = time.time()

	deformParams = (20, 3, [-50, 50, -50, 50], 'firstOrder')
	(p, rIm) = TaylorMSConvexRegistration(target, base, deformParams, 1)

	print('Total Time elapsed: ', (time.time() - start)/60, ' minutes\n\n')

	msDiff = meanSquaresMetric3D(target, rIm)

	print('Deformation Coefficients: ', p, '\n\n')
	print('Registration Parameters: ', deformParams, '\n\n')
	print('Mean Squares Difference: ', msDiff, '\n\n')

	pilIm = Image.fromarray(np.uint8(rIm))
	pilIm.save('/Users/brknight/Documents/ConvexOptimization/figures/MiddleburryEx.png')


def webGLwarpedCatExample():
	basePath = 'images/webGLcat.jpeg'
	targetPath = 'images/warpedCat.jpeg'

	base = readImage(basePath, (300, 300))
	target = readImage(targetPath, (300, 300))

	start = time.time()

	deformParams = (35, 3, [], 'gaussian')
	(p, rIm) = TaylorMSConvexRegistration(target, base, deformParams, 1)

	print('Total Time elapsed: ', (time.time() - start)/60, ' minutes\n\n')

	msDiff = meanSquaresMetric3D(target, rIm)

	print('Deformation Coefficients: ', p, '\n\n')
	print('Registration Parameters: ', deformParams, '\n\n')
	print('Mean Squares Difference: ', msDiff, '\n\n')

	pilIm = Image.fromarray(np.uint8(rIm))
	pilIm.save('/Users/brknight/Documents/ConvexOptimization/figures/webGLwarpedCatEx.png')

def webGLwarpedriceExample():
	basePath = 'images/webGLrice.png'
	targetPath = 'images/webGLwarpedRice.png'

	base = readImage(basePath, (300, 300))
	target = readImage(targetPath, (300, 300))

	start = time.time()

	deformParams = (35, 3, [], 'gaussian')
	(p, rIm) = cvxTaylorMSConvexRegistration(target, base, deformParams, 1)

	print('Total Time elapsed: ', (time.time() - start)/60, ' minutes\n\n')

	msDiff = meanSquaresMetric3D(target, rIm)

	print('Deformation Coefficients: ', p, '\n\n')
	print('Registration Parameters: ', deformParams, '\n\n')
	print('Mean Squares Difference: ', msDiff, '\n\n')

	pilIm = Image.fromarray(np.uint8(rIm))
	pilIm.save('/Users/brknight/Documents/ConvexOptimization/figures/webGLwarpedRiceEx.png')

def main():

	BrainT1ShiftedRotatedEx()


if __name__ == '__main__':
	main()