# # import cv2
# # import sys
# #
# # cascPath = sys.argv[1]
# # faceCascade = cv2.CascadeClassifier(cascPath)
# #
# # video_capture = cv2.VideoCapture(0)
# #
# # while True:
# #     # Capture frame-by-frame
# #     ret, frame = video_capture.read()
# #
# #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #
# #     faces = faceCascade.detectMultiScale(
# #         gray,
# #         scaleFactor=1.1,
# #         minNeighbors=5,
# #         minSize=(30, 30),
# #         flags=cv2.cv.CV_HAAR_SCALE_IMAGE
# #     )
# #
# #     # Draw a rectangle around the faces
# #     for (x, y, w, h) in faces:
# #         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
# #
# #     # Display the resulting frame
# #     cv2.imshow('Video', frame)
# #
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break
# #
# # # When everything is done, release the capture
# # video_capture.release()
# # cv2.destroyAllWindows()
# #
# #
# # cascPath = sys.argv[1]
# # faceCascade = cv2.CascadeClassifier(cascPath)
#
#
# import numpy as np
# import cv2
# # import matplotlib.pyplot as plt
#
# img = cv2.imread('im1.jpg', cv2.IMREAD_GRAYSCALE)
#
# # cv2.imshow('image', img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
#
# import cv2
#
# cap = cv2.VideoCapture('vtest.avi')
#
# while(cap.isOpened()):
#     ret, frame = cap.read()
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#
# # cap.release()
# # out.release()
# # cv2.destroyAllWindows()
# # cv2.destroyAllWindows()
#


import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
img = cv2.imread('im1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

