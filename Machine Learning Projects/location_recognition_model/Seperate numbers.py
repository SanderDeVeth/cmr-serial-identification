import cv2
import numpy as np

# Load the image
image = cv2.imread('input2.png')

# convert to grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# smooth the image to avoid noises
gray = cv2.medianBlur(gray,5)

# Apply adaptive threshold
thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
thresh_color = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)

# apply some dilation and erosion to join the gaps
thresh = cv2.dilate(thresh,None,iterations = 3)
thresh = cv2.erode(thresh,None,iterations = 2)

# Find the contours
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

# For each contour, find the bounding rectangle and draw it

ROI_number = 0
for c in contours:
    area = cv2.contourArea(c)
    if area > 10:
        x,y,w,h = cv2.boundingRect(c)
        ROI = 255 - image[y:y+h, x:x+w]
        ROI = cv2.resize(ROI, (28,28), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 1)
        ROI_number += 1
    