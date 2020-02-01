import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal


def readImage(filename, *args):
	"""Takes in a path to image of any type
    and returns a numpy array of an RGB version of the image, with optional tuple argument to specify size of output image.
    e.g. readImage('myimages/imageofsomething.png', (100, 100)) will return a numpy array of shape (100, 100, 3) 
    corresponding to imageofsomething.png. This is the default input reader for images for this library.
    """

	image = Image.open(filename)
	(m, n) = image.size

	if image.mode != 'RGB':
		image = image.convert('RGB')

	if len(args) == 1:
		(xDim, yDim) = args[0]
	
		if m < xDim:
			xDim = m

		if n < yDim:
			yDim = n
	else:
		(xDim, yDim) = (m,n)

	resizeImage = image.resize((xDim, yDim))
	pixels = resizeImage.load()

	array = []

	for i in range(xDim):
		curRow = []
		for j in range(yDim):
			curPix = pixels[i, j]
			r = curPix[0]
			g = curPix[1]
			b = curPix[2]
			curRow.append([r, g, b])

		array.append(curRow)

	# return np.invert(np.array(array).transpose((1, 0, 2)))
	return np.array(array).transpose((1, 0, 2))



