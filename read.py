import cv2 as cv

img = cv.imread('Photos/cat.jpg')
cv.imshow('Cat', img)


cv.waitKey(0)
#THIS IS FOR CAMERA
# capture = cv.VideoCapture(0)
# while True:
#     isTrue, frame = capture.read()
#     cv.imshow('Video', frame)
#     if cv.waitKey(20) and 0xFF==ord('d'):
#         break
# capture.release()
# cv.destroyAllWindows()