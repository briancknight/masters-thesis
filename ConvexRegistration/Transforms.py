import SimpleITK as sitk
import numpy as np

def display(img):

	nda = sitk.GetArrayViewFromImage(img)
	plt.figure()
	plt.imshow(nda.astype(int))
	plt.show()


def affineTransform(im, xShift, yShift, theta, xScale, yScale, shearX, shearY):

	sitkIm = sitk.GetImageFromArray(im, sitk.sitkVectorFloat32)
	center = sitkIm.TransformContinuousIndexToPhysicalPoint(
		np.array(sitkIm.GetSize())/2.0)

	transform = sitk.AffineTransform(2) # it wants dimension = 2
	transform.SetTranslation((-xShift, yShift))
	transform.Rotate(axis1=0, axis2=1, angle=-theta)
	transform.Scale((1.0/xScale, 1.0/yScale))
	transform.SetCenter(center)
	transform.Shear(axis1=1, axis2=0, coef=shearX)
	transform.Shear(axis1=0, axis2=1, coef=shearY)

	resampler = sitk.ResampleImageFilter()
	resampler.SetReferenceImage(sitkIm)
	resampler.SetInterpolator(sitk.sitkLinear)
	resampler.SetDefaultPixelValue(0)
	resampler.SetTransform(transform)

	outimgsitk = resampler.Execute(sitkIm)

	outimg = sitk.GetArrayFromImage(outimgsitk)
	outimg = outimg.astype(int)

	return outimg


def transformImage(im, transform, *args):

	if len(args) > 0:
		default = args[0]
	else:
		default = 100
	r = transformComponent(im, transform, 0, default)
	g = transformComponent(im, transform, 1, default)
	b = transformComponent(im, transform, 2, default)

	return pieceRGB(r, g, b)


def transformComponent(im, transform, rgb, *args):

	im = sitk.GetImageFromArray(im, sitk.sitkVectorFloat32)
	im = sitk.VectorIndexSelectionCast(im, rgb, sitk.sitkFloat32)

	if len(args) > 0:
		default = args[0]
	else:
		default = 100

	resampler = sitk.ResampleImageFilter()
	resampler.SetReferenceImage(im)
	resampler.SetInterpolator(sitk.sitkLinear)
	resampler.SetDefaultPixelValue(default)
	resampler.SetTransform(transform)

	out = resampler.Execute(im)
	outIm = sitk.GetArrayFromImage(out)

	return outIm.astype(int)


def rotateImage(im, theta):

	return affineTransform(im, 0.0, 0.0, theta, 1.0, 1.0, 0, 0)


def translateImage(im, xCoord, yCoord):

	return affineTransform(im, xCoord, yCoord, 0.0, 1.0, 1.0, 0.0, 0.0)

def scaleImage(im, scaleX, scaleY):

	return affineTransform(im, 0.0, 0.0, 0.0, scaleX, scaleY, 0.0, 0.0)

def shearImage(im, shearX, shearY):

	return affineTransform(im, 0.0, 0.0, 0.0, 1.0, 1.0, shearX, shearY)

def pieceRGB(r, g, b):

	r = np.expand_dims(r, axis=2)
	g = np.expand_dims(g, axis=2)
	b = np.expand_dims(b, axis=2)

	return np.concatenate((r, g, b), axis=2)