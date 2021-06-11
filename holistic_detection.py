import cv2
import mediapipe as mp
import os

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()
def distance(x0, y0, x1, y1):
    return ((((x1-x0)**2) + ((y1-y0)**2))**0.5)
def findUpperLeft(coords, image):
    result = float('inf')
    x = 0.0
    y = 0.0
    for x1,y1 in coords:
        #print(x1,y1)
        dist = distance(0,0,x1,y1)
        if dist < result:
            result = dist
            x = x1; y = y1
    if x > image.shape[1]:
        x = image.shape[1]
    if y > image.shape[0]:
        y = image.shape[0]
    return (x,y)
def findLowerRight(coords, upperLeft, image):
    result = float('-inf')
    x2 = upperLeft[0]
    y2 = upperLeft[1]
    x = 0.0
    y = 0.0
    for x1,y1 in coords:
        #print(x1,y1)
        
        area = (x1-x2)*(y1-y2)
        if area > result:
            
            result = area
            x = x1; y = y1
    #print("max distance: ",result, x, y)
    if x > image.shape[1]:
        x = image.shape[1]
    if y > image.shape[0]:
        y = image.shape[0]
    # print(x, x2, x-x2, y, y2, y-y2)
    # print("max area: ", (x,y), result)
    return (x,y)
def getLandmarks(results):
    people = []
    if results.face_landmarks is not None:
        people.append(results.face_landmarks.landmark)
    if results.left_hand_landmarks is not None:
        people.append(results.left_hand_landmarks.landmark)
    if results.right_hand_landmarks is not None:
        people.append(results.right_hand_landmarks.landmark)
    if results.pose_landmarks is not None:
        people.append(results.pose_landmarks.landmark)
    
    return people
def findPersonCoords(image, results):
    people = getLandmarks(results)
    personCoords = []
    for attribute in people:
        for id, lm in enumerate(attribute):
            ih, iw, ic = image.shape
            x,y = abs(int(lm.x*iw)), abs(int(lm.y*ih))
            personCoords.append([x,y])
    return personCoords


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# For static images:
imagePath = "Photos/photography_project/"
names = os.listdir(imagePath)
IMAGE_FILES = names
with mp_holistic.Holistic(
    static_image_mode=True,
    model_complexity=1) as holistic:
  for file in IMAGE_FILES:
    image = cv2.imread(os.path.join(imagePath, file))

    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    annotated_image = image.copy()
    # mp_drawing.draw_landmarks(
    #     annotated_image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
    # mp_drawing.draw_landmarks(
    #     annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # mp_drawing.draw_landmarks(
    #     annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # mp_drawing.draw_landmarks(
    #     annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    # cv2.imshow("Image", annotated_image)
    
    person = findPersonCoords(image, results)
    cv2.waitKey(0)
    upperLeft = findUpperLeft(person, image)
    lowerRight = findLowerRight(person, upperLeft, image)
    coords = (upperLeft, lowerRight)

    if len(coords) > 0:
        gray = cv2.cvtColor(image[coords[0][1]:coords[1][1] , coords[0][0]:coords[1][0]], cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)
        cv2.rectangle(image, coords[0], coords[1], (0,255,0), 3)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)


    text = "How Focused (higher is better)"
    cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    savepath = "./Photos/temp/"
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    if fm > 100:
        string = savepath+"Score--"+str(int(fm))+"--"+file
        print(string)
        cv2.imwrite(string, annotated_image)

