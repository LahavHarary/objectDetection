from ShapeDetector import *
import imutils
import cv2
import numpy as np

# load the image and resize it to a smalle rfactor so that the shapes can be approximated better
def detectShapes(originalImage):
	newImage = originalImage.copy()
	kernel = np.ones((5, 5), np.uint8)
	gradientImage = cv2.morphologyEx(newImage, cv2.MORPH_GRADIENT, kernel)

	resized = imutils.resize(gradientImage, width=1000)
	ratio = newImage.shape[0] / float(resized.shape[0])

	# convert the resized newImage to grayscale, blur it slightly and threshold it
	gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	_, thresh = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)

	# find contours in the threshold newImage and initialize the shape detector
	contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = imutils.grab_contours(contours)
	shapeDetectorObject = ShapeDetector()

	amTriangle = 0
	amRectangle = 0
	amSquare = 0
	amCircle = 0
	amStar = 0

	# loop over the contours
	for current in contours:
		# compute the center of the contour, then detect the name of the
		# shape using only the contour
		M = cv2.moments(current)
		cX = int((M["m10"] / M["m00"]) * ratio)
		cY = int((M["m01"] / M["m00"]) * ratio)

		# Check which shape came back and add to amount variable
		shape = shapeDetectorObject.detect(current)

		if (shape == "triangle"):
			amTriangle += 1
		elif (shape == "rectangle"):
			amRectangle += 1
		elif (shape == "square"):
			amSquare += 1
		elif (shape == "star"):
			amStar += 1
		elif (shape == "circle"):
			amCircle += 1

		# multiply the contour (x, y)-coordinates by the resize ratio,
		# then draw the contours and the name of the shape on the newImage
		current = current.astype("float")
		current *= ratio
		current = current.astype("int")
		cv2.drawContours(newImage, [current], -1, (0, 255, 0), 3)
		cv2.putText(newImage, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

	# No need for post process: Example why:
	# laplacian = cv2.Laplacian(newImage, cv2.CV_16S, ksize=3)
	# newImage = newImage + laplacian


	return gradientImage,thresh,newImage,amTriangle,amRectangle,amSquare,amStar,amCircle



