import cv2 as cv

img = cv.imread('Photos/cat.jpg')
cv.imshow('Cat', img)


#convert to gray
grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow("gray", grayImage)


#blur (to get rid of noise)
blurredImage = cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT)
#cv.imshow("blur", blurredImage)


#edge cascade
canny = cv.Canny(img, 125, 175)
cv.imshow("canny", canny)


#Dialating image
dialated = cv.dilate(canny, (3,3), iterations=3)
cv.imshow("dialated", dialated)

#eroding / UNDOING dialating to Cascade
eroded = cv.erode(dialated, (3,3), iterations=1)
cv.imshow("Eroded", eroded)


#resize
resized = cv.resize(img, (500,500), interpolation=cv.INTER_AREA) #use INTER_CUBIC or INTER_ LNIEAR for making image bigger not smaller
#cv.imshow('resized', resized)

#cropping
cropped = img[50:200, 200:400]
cv.imshow("cropped", cropped)
cv.waitKey(0)