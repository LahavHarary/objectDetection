import cv2

class ShapeDetector:
    def __init__(self):
        pass

    def detect(self,c):
        shape = "unidentified"
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"

        # if the shape has 4 vertices, it is either a square or a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the bounding box to compute the aspect ratio
            x,y,w,h = cv2.boundingRect(approx)
            ar = w / float(h)

            # a square will have an aspect ratio that is approximately equal to one, otherwise, the shape is rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"


        # if the shape is a start, it will have 10 vertices
        elif len(approx) == 10:
            shape = "star"

        # otherwise we will assume that the shape is circle
        else:
            area = cv2.contourArea(c)
            if(len(approx) >= 6) & (len(approx) <= 23) and area > 30:
                shape = "circle"

        #return the name of the shape
        return shape
