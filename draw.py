import cv2 as cv
import numpy as np



blank = np.zeros((500,500,3), dtype='uint8')
cv.imshow('Blank', blank)

# #1. Paint image a color
# startX = 200
# startY = 300
# endX = 300
# endY = 400
# blank[startX:startY, endX:endY] = 0,255,0
# cv.imshow('Green', blank)


# #2. Draw a rectangle
# cv.rectangle(blank, (0,0), (250,250), (0,255,0), thickness=2)
# cv.imshow('Rectangle', blank)


#3. WRITE TEXT!!!
#          image    text      origin        font                scale  color   thickness
cv.putText(blank, 'Hello', (225, 225), cv.FONT_HERSHEY_TRIPLEX, 0.5, (0,255,0), 1)
cv.imshow('Text', blank)

# img = cv.imread('Photos/cat.jpg')
# cv.imshow('Cat', img)

cv.waitKey(0)