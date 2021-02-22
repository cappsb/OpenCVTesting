import cv2 as cv
# path ="C:/Users/Personal/Downloads/black dot.jpg" 
path ="Photos/black-dot1.jpg"
path ="Photos/fur_elise.jpg"
gray = cv.imread(path, 0) 
# threshold 
th, threshed = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU) 
# findcontours 
cnts = cv.findContours(threshed, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[-2] 

s1 = 3
s2 = 20
xcnts = [] 
for cnt in cnts: 
    if s1<cv.contourArea(cnt) <s2: 
        xcnts.append(cnt) 
print("\nDots number: {}".format(len(xcnts))) 