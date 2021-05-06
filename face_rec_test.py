import face_recognition
import os
import cv2
import sys

def main():
    #detector = FaceMeshDetector(minTrackConfidnce = 0.333)
    imagePath = "Photos/photography_project/"
    names = os.listdir(imagePath)
    print(names)

    for i in range(len(names)):
        #print(type(img))
        img = face_recognition.load_image_file(os.path.join(imagePath, names[i]))
        locations = face_recognition.face_locations(img)
        encodings = face_recognition.face_encodings(img, locations)
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        for face_encoding, face_location in zip(encodings, locations):
            top_left = (face_location[3], face_location[0])
            bottom_right =  (face_location[1], face_location[2])
            color = [0,255,0]

            cv2.rectangle(image, top_left, bottom_right, color, 3)
        cv2.imshow("First Image", image)
        cv2.waitKey(0)

        #print(faces)

if __name__ == "__main__":
    main()