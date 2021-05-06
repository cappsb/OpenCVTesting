import cv2
import mediapipe as mp
import sys
import os, os.path
class FaceMeshDetector():
    def __init__(self, staticMode = False, maxFaces = 2, minDefectionConfidence = 1.0, minTrackConfidnce = 0.333):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDefectionConfidence = minDefectionConfidence
        self.minTrackConfidnce = minTrackConfidnce
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(staticMode, self.maxFaces, 
                                                self.minDefectionConfidence, 
                                                self.minTrackConfidnce)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=2, circle_radius=2)
    def findFaceMesh(self,img, draw =True):
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
        cv2.imshow("First Image", img)
        cv2.waitKey(0)
        return img, faces

def main():
    detector = FaceMeshDetector(minTrackConfidnce = 0.333)
    imagePath = sys.argv[1]
    names = os.listdir(imagePath)
    print(names)

    for i in range( len(names)):
        #print(type(img))
        img = cv2.imread(os.path.join(imagePath, names[i]))
        img, faces = detector.findFaceMesh(img)
        #print(faces)


if __name__ == "__main__":
    main()