import time

start = time.time()

import argparse
import cv2
import os
import pickle

import numpy as np
np.set_printoptions(precision=2)
from sklearn.mixture import GMM
import openface

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')


def getRep(bgrImg):
    # start = time.time()
    # bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image/frame")

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    # if args.verbose:
    #     print("  + Original size: {}".format(rgbImg.shape))
    # if args.verbose:
    #     print("Loading the image took {} seconds.".format(time.time() - start))

    # start = time.time()

    # Get the largest face bounding box
    # bb = align.getLargestFaceBoundingBox(rgbImg) #Bounding box

    # Get all bounding boxes
    bb = align.getAllFaceBoundingBoxes(rgbImg)
    bbs = [bb]
    if bb is None:
        raise Exception("Unable to find a face")
        # raise Exception("Unable to find a face: {}".format(imgPath))
        # return None
    # if args.verbose:
    #     print("Face detection took {} seconds.".format(time.time() - start))

    # start = time.time()

    # alignedFaces = []
    reps = []
    for box in bbs:
        alignedFace = align.align(
                96,
                rgbImg,
                box,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        if alignedFace is None:
            raise Exception("Unable to align the frame")
        # if args.verbose:
        #     print("Alignment took {} seconds.".format(time.time() - start))

        # start = time.time()

        rep = net.forward(alignedFace)
        reps.append((bb.center().x, rep))
        for alignedFace in alignedFaces:
            reps.append(net.forward(alignedFace))

        # if args.verbose:
        #     print("Neural network forward pass took {} seconds.".format(
        #         time.time() - start))

        # print reps
    sreps = sorted(reps, key=lambda x: x[0])
    return sreps


def infer(img,classifierModel):
    with open(classifierModel, 'r') as f:
        (le, clf) = pickle.load(f)  # le - label and clf - classifer

    reps = getRep(img)
    # persons = []
    # confidences = []
    for re in reps:
        try:
            rep = re[1].reshape(1, -1)
            # bbx = re[0]
        except:
            print "No Face detected"
            return (None, None)
        predictions = clf.predict_proba(rep).ravel()
        # print predictions
        maxI = np.argmax(predictions)
        # max2 = np.argsort(predictions)[-3:][::-1][1]
        persons.append(le.inverse_transform(maxI))
        # print str(le.inverse_transform(max2)) + ": "+str( predictions [max2])
        # ^ prints the second prediction
        confidences.append(predictions[maxI])
        # if args.verbose:
        #     print("Prediction took {} seconds.".format(time.time() - start))
        #     pass
        # print("Predict {} with {:.2f} confidence.".format(person, confidence))
        # if isinstance(clf, GMM):
        #     dist = np.linalg.norm(rep - clf.means_[maxI])
        #     print("  + Distance from the mean: {}".format(dist))
        #     pass
    return (persons, confidences)


if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--dlibFacePredictor',
    #     type=str,
    #     help="Path to dlib's face predictor.",
    #     default=os.path.join(
    #         dlibModelDir,
    #         "shape_predictor_68_face_landmarks.dat"))
    # parser.add_argument(
    #     '--networkModel',
    #     type=str,
    #     help="Path to Torch network model.",
    #     default=os.path.join(
    #         openfaceModelDir,
    #         'nn4.small2.v1.t7'))
    # parser.add_argument('--imgDim', type=int,
    #                     help="Default image dimension.", default=96)
    # parser.add_argument(
    #     '--captureDevice',
    #     type=int,
    #     default=0,
    #     help='Capture device. 0 for latop webcam and 1 for usb webcam')
    # parser.add_argument('--width', type=int, default=320)
    # parser.add_argument('--height', type=int, default=240)
    # parser.add_argument('--threshold', type=float, default=0.5)
    # parser.add_argument('--cuda', action='store_true')
    # parser.add_argument('--verbose', action='store_true')
    # parser.add_argument(
    #     'classifierModel',
    #     type=str,
    #     help='The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.')

    # args = parser.parse_args()

    # align = openface.AlignDlib(os.path.join(dlibModelDir,"shape_predictor_68_face_landmarks.dat"))
    align = openface.AlignDlib('/home/fissal/openface/models/dlib/shape_predictor_68_face_landmarks.dat')
    net = openface.TorchNeuralNet(
        os.path.join(openfaceModelDir,'nn4.small2.v1.t7'),
        imgDim=96,
        cuda=False)

    # Capture device. Usually 0 will be webcam and 1 will be usb cam.
    video_capture = cv2.VideoCapture(0)
    video_capture.set(3, 320)
    video_capture.set(4, 240)

    confidenceList = []
    while True:
        ret, frame = video_capture.read()
        persons, confidences = infer(frame, '/home/fissal/openface/generated-embeddings01/classifier.pkl')
        print "P: " + str(persons) + " C: " + str(confidences)
        try:
            # append with two floating point precision
            confidenceList.append('%.2f' % confidences[0])
        except:
            # If there is no face detected, confidences matrix will be empty.
            # We can simply ignore it.
            pass

        for i, c in enumerate(confidences):
            if c <= 0.5:  # 0.5 is kept as threshold for known face.
                persons[i] = "_unknown"

                # Print the person name and conf value on the frame
        cv2.putText(frame, "P: {} C: {}".format(persons, confidences),
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow('', frame)
        # quit the program on the press of key 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
