import cv2
import mediapipe as mp
import sys
import os, os.path

imagePath = sys.argv[1]
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=5, min_detection_confidence = 1, min_tracking_confidence=0.333)
names = os.listdir(imagePath)
print(names)

for i in range( len(names)):
    #print(type(img))
    img = cv2.imread(os.path.join(imagePath, names[i]))
    # cv2.imshow("First Image", img)
    # cv2.waitKey(0)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACE_CONNECTIONS)
    cv2.imshow("First Image", img)
    cv2.waitKey(0)
