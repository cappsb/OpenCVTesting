import cv2
import time
import poseDetect as pm
import os

detector = pm.PoseDetector()

imagePath = "Photos/photography_project/"
names = os.listdir(imagePath)
print(names)

for i in range( len(names)):
    #print(type(img))
    img = cv2.imread(os.path.join(imagePath, names[i]))
    img = detector.findPose(img)
    lmList = detector.getPosition(img)
    print(lmList)

    cv2.imshow("Image", img)
    cv2.waitKey(0)