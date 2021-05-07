import cv2
import mediapipe as mp
import sys
import os, os.path
import face_recognition
class FaceMeshDetector():
    def __init__(self, staticMode = False, maxFaces = 2, minDefectionConfidence = 0.01, minTrackConfidnce = 0.015):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDefectionConfidence = minDefectionConfidence
        self.minTrackConfidnce = minTrackConfidnce
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode = staticMode, max_num_faces = self.maxFaces, 
                                                min_detection_confidence = self.minDefectionConfidence, 
                                                min_tracking_confidence = self.minTrackConfidnce)

        self.drawSpec = self.mpDraw.DrawingSpec(thickness=0.5, circle_radius=0.5)
    def findFaceMesh(self,img, name, draw =True, useFaceRec = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS)
                faceCoords = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x,y = int(lm.x*iw), int(lm.y*ih)
                    faceCoords.append([x,y])

                faces.append(faceCoords)  
        if useFaceRec:
            imagePath = "Photos/photography_project/"
            names = os.listdir(imagePath)
            image = face_recognition.load_image_file(os.path.join(imagePath, name))
            locations = face_recognition.face_locations(image)
            encodings = face_recognition.face_encodings(image, locations)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            for face_encoding, face_location in zip(encodings, locations):
                top_left = (face_location[3], face_location[0])
                bottom_right =  (face_location[1], face_location[2])
                color = [0,255,0]

                cv2.rectangle(image, top_left, bottom_right, color, 3)      
            cv2.imshow("First Image", image)
        if draw:
            cv2.imshow("Mesh", img)
            cv2.waitKey(0)
        return img, faces

def main():
    detector = FaceMeshDetector()
    imagePath = "Photos/photography_project/"
    names = os.listdir(imagePath)
    print(names)

    for i in range( len(names)):
        #print(type(img))
        img = cv2.imread(os.path.join(imagePath, names[i]))
        img, faces = detector.findFaceMesh(img, names[i])
        #print(faces)

if __name__ == "__main__":
    main()