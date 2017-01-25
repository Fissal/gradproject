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

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('cascade.xml')
cap = cv2.VideoCapture(0)

while (True):
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print gray
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print faces
    # if (len(faces) != 0):
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 2)

    cv2.imshow('frame', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#

# import cv2
#
# vidcap = cv2.VideoCapture('vtest.mp4')
# vidcap.set(cv2.CAP_PROP_POS_MSEC,20000)      # just cue to 20 sec. position
# success,image = vidcap.read()
# if success:
#     print "s"
#     cv2.imwrite("frame20sec.jpg", image)     # save frame as JPEG file
#     cv2.imshow("20sec",image)
#     cv2.waitKey()




