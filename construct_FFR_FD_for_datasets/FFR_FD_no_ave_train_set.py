# -*- coding: UTF-8 -*-


import cv2
import numpy as np
import argparse
from imutils import face_utils
import dlib
import operator
from functools import reduce
import os


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="choose datasets: UADFV, DeepfakeTIMIT_HQ, DeepfakeTIMIT_LQ,"
                     "FF++_Deepfakes_c40, FF++_Deepfakes_raw, DFD, DFDC, CelebDF_V2")
args = vars(ap.parse_args())


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')


def saveFeatureSet(featureSet, name):
    fdir = "../FFR_FD_no_ave/train/"
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    fname = fdir + name + '.npy'
    if name[-1] != '_':
        np.save(fname, featureSet)
        print(name + ' have saved')


def siftSta(imgDir, imgFormat):
    imgNum = 0
    for subdir, dirs, files in os.walk(imgDir):
        for file in files:
            if file[-4:] == imgFormat:
                imgNum += 1
    FFR_FD = np.zeros(shape=(imgNum, 1024))

    for subdir, dirs, files in os.walk(imgDir):
        num = 0
        for file in files:
            if file[-4:] == imgFormat:

                img = cv2.imread(subdir + '\\' + file)
                sift = cv2.xfeatures2d.SIFT_create()
                kp, des = sift.detectAndCompute(img, None)
                if len(kp) == 0:
                    FFR_FD[num] = np.zeros(1024, dtype=float)
                    num += 1
                    continue
                desFuse = reduce(operator.add, des)

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rects = detector(gray, 1)
                existFaceReg = False
                for (i, rect) in enumerate(rects):
                    if existFaceReg:
                        break
                    existFaceReg = True
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    for (name, (j, k)) in face_utils.FACIAL_LANDMARKS_IDXS.items():

                        if name == 'jaw':
                            break
                        (x, y, w, h) = cv2.boundingRect(np.array([shape[j:k]]))
                        roi = img[y - int(h / 10): y + int(1.1 * h), x - int(w / 10):x + int(1.1 * w)]

                        if roi.shape[0] == 0 or roi.shape[1] == 0:
                            desNone = np.zeros(128, dtype=float)
                            desFuse = np.append(desFuse, desNone)
                            continue

                        FDr = np.float64([]).reshape(0, 128)
                        i = -1
                        for ak in kp:
                            i += 1
                            if ((y - h / 10) <= ak.pt[1] <= (y + 1.1 * h)) and (
                                    (x - w / 10) <= ak.pt[0] <= (x + 1.1 * w)):
                                FDr = np.append(FDr, des[i, :][np.newaxis, :], axis=0)
                        if FDr.shape[0] != 0:
                            FDr = reduce(operator.add, FDr)
                            desFuse = np.append(desFuse, FDr)
                        else:
                            FDr = np.zeros(128, dtype=float)
                            desFuse = np.append(desFuse, FDr)

                if not existFaceReg:
                    desFuse = np.append(desFuse, np.zeros(7 * 128, dtype=float))
                FFR_FD[num] = desFuse
                num += 1

        saveFeatureSet(FFR_FD, name=args['dataset'] + '_' + subdir.split('/')[-2] + '_sift_FFR_FD')


def surfSta(imgDir, imgFormat):
    imgNum = 0
    for subdir, dirs, files in os.walk(imgDir):
        for file in files:
            if file[-4:] == imgFormat:
                imgNum += 1
    FFR_FD = np.zeros(shape=(imgNum, 512))

    for subdir, dirs, files in os.walk(imgDir):
        num = 0
        for file in files:
            if file[-4:] == imgFormat:

                img = cv2.imread(subdir + '\\' + file)
                surf = cv2.xfeatures2d.SURF_create()
                kp, des = surf.detectAndCompute(img, None)
                if len(kp) == 0:
                    FFR_FD[num] = np.zeros(512, dtype=float)
                    num += 1
                    continue
                desFuse = reduce(operator.add, des)

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rects = detector(gray, 1)
                existFaceReg = False
                for (i, rect) in enumerate(rects):
                    if existFaceReg:
                        break
                    existFaceReg = True
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    for (name, (j, k)) in face_utils.FACIAL_LANDMARKS_IDXS.items():

                        if name == 'jaw':
                            break
                        (x, y, w, h) = cv2.boundingRect(np.array([shape[j:k]]))
                        roi = img[y - int(h / 10): y + int(1.1 * h), x - int(w / 10):x + int(1.1 * w)]

                        if roi.shape[0] == 0 or roi.shape[1] == 0:
                            desNone = np.zeros(64, dtype=float)
                            desFuse = np.append(desFuse, desNone)
                            continue

                        FDr = np.float64([]).reshape(0, 64)
                        i = -1
                        for ak in kp:
                            i += 1
                            if ((y - h / 10) <= ak.pt[1] <= (y + 1.1 * h)) and (
                                    (x - w / 10) <= ak.pt[0] <= (x + 1.1 * w)):
                                FDr = np.append(FDr, des[i, :][np.newaxis, :], axis=0)
                        if FDr.shape[0] != 0:
                            FDr = reduce(operator.add, FDr)
                            desFuse = np.append(desFuse, FDr)
                        else:
                            FDr = np.zeros(64, dtype=float)
                            desFuse = np.append(desFuse, FDr)

                if not existFaceReg:
                    desFuse = np.append(desFuse, np.zeros(7 * 64, dtype=float))
                FFR_FD[num] = desFuse
                num += 1

        saveFeatureSet(FFR_FD, name=args['dataset'] + '_' + subdir.split('/')[-2] + '_surf_FFR_FD')


def orbSta(imgDir, imgFormat):
    imgNum = 0
    for subdir, dirs, files in os.walk(imgDir):
        for file in files:
            if file[-4:] == imgFormat:
                imgNum += 1
    FFR_FD = np.zeros(shape=(imgNum, 256))

    for subdir, dirs, files in os.walk(imgDir):
        num = 0
        for file in files:
            if file[-4:] == imgFormat:

                img = cv2.imread(subdir + '\\' + file)
                orb = cv2.ORB_create()
                kp, des = orb.detectAndCompute(img, None)
                if len(kp) == 0:
                    FFR_FD[num] = np.zeros(256, dtype=float)
                    num += 1
                    continue
                desFuse = reduce(operator.add, des)

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rects = detector(gray, 1)
                existFaceReg = False
                for (i, rect) in enumerate(rects):
                    if existFaceReg:
                        break
                    existFaceReg = True
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    for (name, (j, k)) in face_utils.FACIAL_LANDMARKS_IDXS.items():

                        if name == 'jaw':
                            break
                        (x, y, w, h) = cv2.boundingRect(np.array([shape[j:k]]))
                        roi = img[y - int(h / 10): y + int(1.1 * h), x - int(w / 10):x + int(1.1 * w)]

                        if roi.shape[0] == 0 or roi.shape[1] == 0:
                            desNone = np.zeros(32, dtype=float)
                            desFuse = np.append(desFuse, desNone)
                            continue

                        FDr = np.float32([]).reshape(0, 32)
                        i = -1
                        for ak in kp:
                            i += 1
                            if ((y - h / 10) <= ak.pt[1] <= (y + 1.1 * h)) and (
                                    (x - w / 10) <= ak.pt[0] <= (x + 1.1 * w)):
                                FDr = np.append(FDr, des[i, :][np.newaxis, :], axis=0)
                        if FDr.shape[0] != 0:
                            FDr = reduce(operator.add, FDr)
                            desFuse = np.append(desFuse, FDr)
                        else:
                            FDr = np.zeros(32, dtype=float)
                            desFuse = np.append(desFuse, FDr)

                if not existFaceReg:
                    desFuse = np.append(desFuse, np.zeros(7 * 32, dtype=float))
                FFR_FD[num] = desFuse
                num += 1

        saveFeatureSet(FFR_FD, name=args['dataset'] + '_' + subdir.split('/')[-2] + '_orb_FFR_FD')


def fastSta(imgDir, imgFormat):
    imgNum = 0
    for subdir, dirs, files in os.walk(imgDir):
        for file in files:
            if file[-4:] == imgFormat:
                imgNum += 1
    FFR_FD = np.zeros(shape=(imgNum, 256))

    for subdir, dirs, files in os.walk(imgDir):
        num = 0
        for file in files:
            if file[-4:] == imgFormat:

                img = cv2.imread(subdir + '\\' + file)
                fast = cv2.cv2.FastFeatureDetector_create()
                kp = fast.detect(img, None)
                brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
                kp, des = brief.compute(img, kp)
                if len(kp) == 0:
                    FFR_FD[num] = np.zeros(256, dtype=float)
                    num += 1
                    continue
                desFuse = reduce(operator.add, des)

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rects = detector(gray, 1)
                existFaceReg = False
                for (i, rect) in enumerate(rects):
                    if existFaceReg:
                        break
                    existFaceReg = True
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    for (name, (j, k)) in face_utils.FACIAL_LANDMARKS_IDXS.items():

                        if name == 'jaw':
                            break
                        (x, y, w, h) = cv2.boundingRect(np.array([shape[j:k]]))
                        roi = img[y - int(h / 10): y + int(1.1 * h), x - int(w / 10):x + int(1.1 * w)]

                        if roi.shape[0] == 0 or roi.shape[1] == 0:
                            desNone = np.zeros(32, dtype=float)
                            desFuse = np.append(desFuse, desNone)
                            continue

                        FDr = np.float32([]).reshape(0, 32)
                        i = -1
                        for ak in kp:
                            i += 1
                            if ((y - h / 10) <= ak.pt[1] <= (y + 1.1 * h)) and (
                                    (x - w / 10) <= ak.pt[0] <= (x + 1.1 * w)):
                                FDr = np.append(FDr, des[i, :][np.newaxis, :], axis=0)
                        if FDr.shape[0] != 0:
                            FDr = reduce(operator.add, FDr)
                            desFuse = np.append(desFuse, FDr)
                        else:
                            FDr = np.zeros(32, dtype=float)
                            desFuse = np.append(desFuse, FDr)

                if not existFaceReg:
                    desFuse = np.append(desFuse, np.zeros(7 * 32, dtype=float))
                FFR_FD[num] = desFuse
                num += 1

        saveFeatureSet(FFR_FD, name=args['dataset'] + '_' + subdir.split('/')[-2] + '_fast_FFR_FD')


def akazeSta(imgDir, imgFormat):
    imgNum = 0
    for subdir, dirs, files in os.walk(imgDir):
        for file in files:
            if file[-4:] == imgFormat:
                imgNum += 1
    FFR_FD = np.zeros(shape=(imgNum, 488))

    for subdir, dirs, files in os.walk(imgDir):
        num = 0
        for file in files:
            if file[-4:] == imgFormat:

                img = cv2.imread(subdir + '\\' + file)
                akaze = cv2.AKAZE_create()
                kp, des = akaze.detectAndCompute(img, None)
                if len(kp) == 0:
                    FFR_FD[num] = np.zeros(488, dtype=float)
                    num += 1
                    continue
                desFuse = reduce(operator.add, des)

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rects = detector(gray, 1)
                existFaceReg = False
                for (i, rect) in enumerate(rects):
                    if existFaceReg:
                        break
                    existFaceReg = True
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    for (name, (j, k)) in face_utils.FACIAL_LANDMARKS_IDXS.items():

                        if name == 'jaw':
                            break
                        (x, y, w, h) = cv2.boundingRect(np.array([shape[j:k]]))
                        roi = img[y - int(h / 10): y + int(1.1 * h), x - int(w / 10):x + int(1.1 * w)]

                        if roi.shape[0] == 0 or roi.shape[1] == 0:
                            desNone = np.zeros(61, dtype=float)
                            desFuse = np.append(desFuse, desNone)
                            continue

                        FDr = np.float64([]).reshape(0, 61)
                        i = -1
                        for ak in kp:
                            i += 1
                            if ((y - h / 10) <= ak.pt[1] <= (y + 1.1 * h)) and (
                                    (x - w / 10) <= ak.pt[0] <= (x + 1.1 * w)):
                                FDr = np.append(FDr, des[i, :][np.newaxis, :], axis=0)
                        if FDr.shape[0] != 0:
                            FDr = reduce(operator.add, FDr)
                            desFuse = np.append(desFuse, FDr)
                        else:
                            FDr = np.zeros(61, dtype=float)
                            desFuse = np.append(desFuse, FDr)

                if not existFaceReg:
                    desFuse = np.append(desFuse, np.zeros(7 * 61, dtype=float))
                FFR_FD[num] = desFuse
                num += 1

        saveFeatureSet(FFR_FD, name=args['dataset'] + '_' + subdir.split('/')[-2] + '_akaze_FFR_FD')


img_format = '.jpg'

base_dir = '../datasets/faces/' + args['dataset'] + '/train/'
real_dir = base_dir + 'real/'
fake_dir = base_dir + 'fake/'
print(args['dataset'] + ':')
siftSta(real_dir, img_format)
siftSta(fake_dir, img_format)
surfSta(real_dir, img_format)
surfSta(fake_dir, img_format)
orbSta(real_dir, img_format)
orbSta(fake_dir, img_format)
fastSta(real_dir, img_format)
fastSta(fake_dir, img_format)
akazeSta(real_dir, img_format)
akazeSta(fake_dir, img_format)
