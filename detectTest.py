import cv2 
import sys


imagePath = sys.argv[1]
cascPath = sys.argv[2]
cascPath2 = sys.argv[2]


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
bodyCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")
leftEyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_lefteye_2splits.xml")
rightEyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_rightteye_2splits.xml")
drewFace = False
print(cv2.data.haarcascades)
for i in range(1,24):
    image = cv2.imread(imagePath+str(i)+".jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("First Image", image)
    cv2.waitKey(0)
    faces = faceCascade.detectMultiScale(
        gray,
        # scaleFactor=1.255,
         scaleFactor=1.05,
        minNeighbors=5,
        # minSize=(30, 30),
        #maxSize = (105,105),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    body = bodyCascade.detectMultiScale(
        gray,
        scaleFactor=1.025,
        minNeighbors=4,
        # minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    # print("Iteration:"+str(i))
    print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = image[y:y+h,x:x+w]
        # eyes = eyeCascade.detectMultiScale(roi_gray, 1.01, 5)
        leftEye = eyeCascade.detectMultiScale(roi_gray, 1.01, 5)
        rightEye = eyeCascade.detectMultiScale(roi_gray, 1.01, 5)

        # for (ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)

        for (xl,yl,wl,hl) in leftEye:
            cv2.rectangle(roi_color, (xl, yl), (xl+wl, yl+hl), (0, 111, 255), 2)

        for (xr,yr,wr,hr) in rightEye:
            cv2.rectangle(roi_color, (xr, yr), (xr+wr, yr+hr), (0, 111, 255), 2)
        drewFace=True
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # cv2.imshow("Faces found", image)
    # if not drewFace:
    for (x,y,w,h) in body:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)



    
    cv2.imshow("Faces found1", image)
    cv2.waitKey(0)


#ALSO USE haarcascade_fullbody.xml AND haarcascade_frontalface_default.xml AND haarcascade_eye.xml