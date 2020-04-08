from Taylor08 import *

def DeformedBrainPDExample():

	basePath = 'images/BrainProtonDensitySlice.png'
	targetPath = 'images/DeformedBrainPD.png'

	base = readImage(basePath)
	target = readImage(targetPath)

	start = time.time()

	deformParams = (30, 3, [-50, 50, -50, 50], 'gaussian')
	(p, rIm) = TaylorMSConvexRegistration(target, base, deformParams, 1)

	print('Total Time elapsed: ', (time.time() - start)/60, ' minutes\n\n')

	print('Deformation Coefficients: ', p, '\n\n')
	print('Registration Parameters: ', deformParams, '\n\n')

	pilIm = Image.fromarray(np.uint8(rIm))
	pilIm.save('/Users/brknight/Documents/ConvexOptimization/figures/DeformedBrainPDEx.png')


def WarpedBrainPDExample():

	basePath = 'images/BrainProtonDensitySlice.png'
	targetPath = 'images/WarpedBrainProtonDensitySlice.png'

	base = readImage(basePath)
	target = readImage(targetPath)

	start = time.time()

	deformParams = (40, 3, [-50, 50, -50, 50], 'gaussian')
	(p, rIm) = cvxTaylorMSConvexRegistration(target, base, deformParams, 1)

	print('Total Time elapsed: ', (time.time() - start)/60, ' minutes\n\n')

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

	deformParams = (20, 3, [-50, 50, -50, 50], 'gaussian')
	(p, rIm) = TaylorMSConvexRegistration(target, base, deformParams, 1)

	print('Total Time elapsed: ', (time.time() - start)/60, ' minutes\n\n')

	print('Deformation Coefficients: ', p, '\n\n')
	print('Registration Parameters: ', deformParams, '\n\n')

	pilIm = Image.fromarray(np.uint8(rIm))
	pilIm.save('/Users/brknight/Documents/ConvexOptimization/figures/BrainMidSaggitalGaussianEx.png')


def BrainMidSaggitalThirdOrderExample():

	basePath = 'images/BrainMidSagittalSlice.png'
	targetPath = 'images/DefMid.jpg'

	base = readImage(basePath, (187, 155))
	target = readImage(targetPath)

	start = time.time()

	deformParams = (30, 3, [-50, 50, -50, 50], 'thirdOrder')
	(p, rIm) = cvxTaylorMSConvexRegistration(target, base, deformParams, 1)

	print('Total Time elapsed: ', (time.time() - start)/60, ' minutes\n\n')

	print('Deformation Coefficients: ', p, '\n\n')
	print('Registration Parameters: ', deformParams, '\n\n')

	pilIm = Image.fromarray(np.uint8(rIm))
	pilIm.save('/Users/brknight/Documents/ConvexOptimization/figures/BrainMidSaggitalThirdOrderEx.png')


def BrainT1ShiftedRotatedEx():

	basePath = 'images/BrainT1Slice.png'
	targetPath = 'images/BrainT1SliceR10X13Y17.png'
	base = readImage(basePath)
	target = readImage(targetPath)

	start = time.time()

	deformParams = (30, 3, [], 'firstOrder')
	(p, rIm) = cvxTaylorMSConvexRegistration(target, base, deformParams, 1)

	print('Total Time elapsed: ', (time.time() - start)/60, ' minutes\n\n')

	print('Deformation Coefficients: ', p, '\n\n')
	print('Registration Parameters: ', deformParams, '\n\n')

	pilIm = Image.fromarray(np.uint8(rIm))
	pilIm.save('/Users/brknight/Documents/ConvexOptimization/figures/Braint1ShiftedRotatedEx.png')


def BrainPDShiftedX13Y17Example():

	basePath = 'images/BrainProtonDensitySlice.png'
	targetPath = 'images/BrainProtonDensitySliceShifted13x17y.png'

	base = readImage(basePath)
	target = readImage(targetPath)

	start = time.time()

	deformParams = (20, 3, [-50, 50, -50, 50], 'firstOrder')
	(p, rIm) = TaylorMSConvexRegistration(target, base, deformParams, 1)

	print('Total Time elapsed: ', (time.time() - start)/60, ' minutes\n\n')

	print('Deformation Coefficients: ', p, '\n\n')
	print('Registration Parameters: ', deformParams, '\n\n')

	pilIm = Image.fromarray(np.uint8(rIm))
	pilIm.save('/Users/brknight/Documents/ConvexOptimization/figures/BrainPDShiftedX13Y17Ex.png')


def RiceDeformedExample():

	basePath = 'images/rice.png'
	targetPath = 'images/DeformedRice.png'

	base = readImage(basePath, (300, 300))
	target = readImage(targetPath)

	start = time.time()

	deformParams = (50, 3, [-50, 50, -50, 50], 'gaussian')
	(p, rIm) = TaylorMSConvexRegistration(target, base, deformParams, 1)

	print('Total Time elapsed: ', (time.time() - start)/60, ' minutes\n\n')

	print('Deformation Coefficients: ', p, '\n\n')
	print('Registration Parameters: ', deformParams, '\n\n')

	pilIm = Image.fromarray(np.uint8(rIm))
	pilIm.save('/Users/brknight/Documents/ConvexOptimization/figures/DeformedRiceEx.png')

def main():

	RiceDeformedExample()


if __name__ == '__main__':
	main()