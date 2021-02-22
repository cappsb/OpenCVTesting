import cv2 as cv

img = cv.imread('Photos/fur_elise.jpg')
cv.imshow('Cat', img)
grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("gray", grayImage)
canny = cv.Canny(img, 125, 175)
cv.imshow("canny", canny)

#TO MAKE AN IMAGE BLACK AND WHITE
ret, th = cv.threshold(grayImage, 125,255, cv.THRESH_BINARY)
cv.imshow("th", th)


# coords    rep                         ret and find      method
contours, hier = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

cv.waitKey(0)