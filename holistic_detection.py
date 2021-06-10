import cv2
import mediapipe as mp
import os
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# For static images:
imagePath = "Photos/photography_project/"
names = os.listdir(imagePath)
IMAGE_FILES = names
print(IMAGE_FILES)
with mp_holistic.Holistic(
    static_image_mode=True,
    model_complexity=2) as holistic:
  for file in range(len(IMAGE_FILES)):
    print(file)
    image = cv2.imread(os.path.join(imagePath, names[file]))
    print(type(image))
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
      print(
          f'Nose coordinates: ('
          f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
          f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
      )
    # Draw pose, left and right hands, and face landmarks on the image.
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
    mp_drawing.draw_landmarks(
        annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(
        annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(
        annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    cv2.imshow("Image", annotated_image)
    cv2.waitKey(0)
    cv2.imwrite('/tmp/annotated_image' + '.png', annotated_image)