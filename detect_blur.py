from imutils import paths
from FaceMeshModule import FaceMeshDetector
import cv2
import sys
import os

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()
def distance(x0, y0, x1, y1):
    return ((((x1-x0)**2) + ((y1-y0)**2))**0.5)
def findUpperLeft(face):
    result = float('inf')
    x = 0.0
    y = 0.0
    for x1,y1 in face:
        #print(x1,y1)
        dist = distance(0,0,x1,y1)
        if dist < result:
            result = dist
            x = x1; y = y1
    #print("min distance: ",result, x, y)
    return (x,y)
def findLowerRight(face):
    result = float('-inf')
    x = 0.0
    y = 0.0
    for x1,y1 in face:
        #print(x1,y1)
        dist = distance(0,0,x1,y1)
        if dist > result:
            result = dist
            x = x1; y = y1
    #print("max distance: ",result, x, y)
    return (x,y)

def main():
    detector = FaceMeshDetector()
    imagePath = "Photos/photography_project/"
    threshold = 255.0
    names = os.listdir(imagePath)
    print(names)

    for i in range( len(names)):
        #print(type(img))
        img = cv2.imread(os.path.join(imagePath, names[i]))
    
        img, faces = detector.findFaceMesh(img, names[i],draw = False, useFaceRec=False)
        coords = []
        for face in faces:
            upperLeft = findUpperLeft(face)
            lowerRight = findLowerRight(face)
            # print(upperLeft)
            # print(lowerRight)
            coords.append((upperLeft, lowerRight))
        #if there are faces detected, then only do blur detection on the face
        #otherwise, do blur detection on the entire image.
        if len(coords) > 0:
            summation = 0
            for coord in coords:
                gray = cv2.cvtColor(img[coord[0][1]:coord[1][1] , coord[0][0]:coord[1][0]], cv2.COLOR_BGR2GRAY)
                summation += variance_of_laplacian(gray)
            fm = summation / (len(coords)+0.0)
            
        
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fm = variance_of_laplacian(gray)
        text = "How Focused (higher is better)"
        cv2.putText(img, "{}: {:.2f}".format(text, fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(0)

if __name__ == '__main__':
    main()