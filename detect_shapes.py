from ShapeDetector import *
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt
import numpy as np

# load the image and resize it to a smalle rfactor so that the shapes can be approximated better
originalImage = cv2.imread("photos/one rectangle two circles.png")
#originalImage = cv2.imread("photos/three rectangles four circles.png")
#originalImage = cv2.imread("photos/three rectangles four circles with paint on top of them.png")
#originalImage = cv2.imread("photos/three rectangles four circles with paint on top of them and on the background.png")
#originalImage = cv2.imread("photos/three rectangles one triangle one circle one star.png")
image = originalImage.copy()

kernel = np.ones((5,5),np.uint8)
gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

resized = imutils.resize(gradient,width=1000)
ratio = image.shape[0] / float(resized.shape[0])

# convert the resized image to grayscale, blur it slightly and threshold it
gray = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray,(5,5),0)
_ ,thresh = cv2.threshold(blurred,25,255,cv2.THRESH_BINARY)

# find contours in the threshold image and initialize the shape detector
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
shapeDetectorObject = ShapeDetector()

amountTriangle = 0
amountRectangle = 0
amountSquare = 0
amountPentagon = 0
amountCircle = 0
amountStar = 0

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
		amountTriangle += 1
	elif(shape == "rectangle"):
		amountRectangle += 1
	elif(shape == "square"):
		amountSquare += 1
	elif (shape == "pentagon"):
		amountPentagon += 1
	elif(shape == "star"):
		amountStar += 1
	elif(shape == "circle"):
		amountCircle += 1

	# multiply the contour (x, y)-coordinates by the resize ratio,
	# then draw the contours and the name of the shape on the image
	current = current.astype("float")
	current *= ratio
	current = current.astype("int")
	cv2.drawContours(image, [current], -1, (0, 255, 0), 3)
	cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# No need for post process: Example why:
#laplacian = cv2.Laplacian(image, cv2.CV_16S, ksize=3)
#image = image + laplacian

# show the output image
plt.figure()

plt.subplot(221)
plt.imshow(originalImage)
plt.title("original image")

plt.subplot(222)
plt.imshow(gradient)
plt.title("gradient image")

plt.subplot(223)
plt.imshow(thresh)
plt.title("thresh")

plt.subplot(224)
plt.imshow(image)
plt.title("image after process")

plt.show()

if(amountTriangle > 0):
	print("Amount of Triangles in the photo is ", amountTriangle)
if(amountRectangle > 0):
	print("Amount of Rectangles in the photo is ", amountRectangle)
if(amountSquare > 0):
	print("Amount of Squares in the photo is ", amountSquare)
if(amountPentagon > 0):
	print("Amount of Pentagons in the photo is ", amountPentagon)
if(amountStar > 0):
	print("Amount of Stars in the photo is ", amountStar)
if(amountCircle > 0):
	print("Amount of Circles in the photo is ", amountCircle)
