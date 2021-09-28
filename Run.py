from detect_shapes import *

#originalImage = cv2.imread("photos/one rectangle two circles.png")
#originalImage = cv2.imread("photos/three rectangles four circles.png")
#originalImage = cv2.imread("photos/three rectangles four circles with paint on top of them.png")
#originalImage = cv2.imread("photos/three rectangles four circles with paint on top of them and on the background.png")
#originalImage = cv2.imread("photos/three rectangles one triangle one circle one star.png")
originalImage = cv2.imread("photos/three circles in rectangle.png")
#originalImage = cv2.imread("photos/four squares with noise.png")


gradientImage, thresh, newImage, \
amTriangle, amRectangle, amSquare, amStar, amCircle = detectShapes(originalImage)

# show the output newImage
plt.figure()

plt.subplot(221)
plt.imshow(originalImage)
plt.title("original image")

plt.subplot(222)
plt.imshow(gradientImage)
plt.title("gradient image")

plt.subplot(223)
plt.imshow(thresh)
plt.title("thresh image")

plt.subplot(224)
plt.imshow(newImage)
plt.title("image after process")

plt.show()

if (amTriangle > 0):
    print("Amount of Triangles in the photo is ", amTriangle)
if (amRectangle > 0):
    print("Amount of Rectangles in the photo is ", amRectangle)
if (amSquare > 0):
    print("Amount of Squares in the photo is ", amSquare)
if (amStar > 0):
    print("Amount of Stars in the photo is ", amStar)
if (amCircle > 0):
    print("Amount of Circles in the photo is ", amCircle)