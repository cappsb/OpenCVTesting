import cv2 as cv
def rescaleFrame(frame, scale=0.75):
    width = int (frame.shape[1]*scale)
    height = int (frame.shape[0]*scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def changeRes(width, height):
    #Only live videos
    capture.set(3, width)
    capture.set(4, height)


img = cv.imread('Photos/cat.jpg')
cv.imshow('Cat', img)
imgResized = rescaleFrame(img)
cv.imshow('Resized', imgResized)
cv.waitKey(0)




# capture = cv.VideoCapture(0)
# while True:
#     isTrue, frame = capture.read()
#     frame_resized = rescaleFrame(frame)
#     cv.imshow('Video', frame)
#     cv.imshow('Resized', frame_resized)
#     if cv.waitKey(20) and 0xFF==ord('d'):
#         break
# capture.release()
# cv.destroyAllWindows()