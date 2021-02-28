import cv2 as cv

def rescaleFrame(frame, scale=0.75):
    width = int (frame.shape[1]*scale)
    height = int (frame.shape[0]*scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
# path ="C:/Users/Personal/Downloads/black dot.jpg" 
image = cv.imread('Photos/dom2.jpg')

imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
imgray = rescaleFrame(imgray, 0.5)
#imgray = ~imgray
#imgray = cv.GaussianBlur(imgray,(7,7),0)
thresh = cv.threshold(imgray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

# Remove horizontal
horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (25,1))
detected_lines = cv.morphologyEx(thresh, cv.MORPH_OPEN, horizontal_kernel, iterations=100)
cnts = cv.findContours(detected_lines, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv.drawContours(image, [c], -1, (255,255,255), 2)


ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
print(type(contours))
s1 = 500
s2 = 900
xconts = [] 
for cont in contours: 
    if s1<cv.contourArea(cont) <s2: 
        #print(cont)
        xconts.append(cont) 

for cont in xconts:
    # print(cont, "happy")
    # print(cont[0][0][0], cont[0][0][1], "sad")
    #print(cv.contourArea(cont))
    M = cv.moments(cont)
    cX = int(M["m10"]/M["m00"])
    cY = int(M["m01"]/M["m00"])
    # draw the contour and center of the shape on the image
    #cv.drawContours(imgray, [cont], -1, (0, 255, 0), 2)
    cv.circle(imgray, (cX, cY), 30, (0, 0, 255), 2)
#cv.drawContours(imgray, contours, -1, (0,255,0), 3)
cv.imshow('gray', imgray)

#print(xconts)
print("\nDots number: {}".format(len(xconts))) 
cv.waitKey(0)