# -*- coding: UTF-8 -*-


from filePath import *
import cv2
import os
import argparse
import sys
import dlib
from imutils import face_utils
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="choose datasets: UADFV, DeepfakeTIMIT_HQ, DeepfakeTIMIT_LQ,"
                     "FF++_Deepfakes_c40, FF++_Deepfakes_raw, DFD, DFDC, CelebDF_V2")
args = vars(ap.parse_args())


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def siftSta(imgDir, imgFormat):
    face_parts_kp = {}
    for subdir, dirs, files in os.walk(imgDir):
        for file in files:
            if file[-4:] == imgFormat:

                img = cv2.imread(subdir + '\\' + file)
                sift = cv2.xfeatures2d.SIFT_create()
                kp, des = sift.detectAndCompute(img, None)
                face_parts_kp.setdefault('entire', []).append(len(kp))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rects = detector(gray, 1)

                for (i, rect) in enumerate(rects):
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    for (name, (j, k)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                        numKp = 0
                        (x, y, w, h) = cv2.boundingRect(np.array([shape[j:k]]))
                        roi = img[y - int(h / 10): y + int(1.1 * h), x - int(w / 10):x + int(1.1 * w)]
                        if roi.shape[0] == 0 or roi.shape[1] == 0:
                            continue
                        for ak in kp:
                            if ((y - h / 10) <= ak.pt[1] <= (y + 1.1 * h)) and ((x - w / 10) <= ak.pt[0] <= (x + 1.1 * w)):
                                numKp += 1
                        face_parts_kp.setdefault(name, []).append(numKp)
    return face_parts_kp


def surfSta(imgDir, imgFormat):
    face_parts_kp = {}
    for subdir, dirs, files in os.walk(imgDir):
        for file in files:
            if file[-4:] == imgFormat:

                img = cv2.imread(subdir + '\\' + file)
                surf = cv2.xfeatures2d.SURF_create()
                kp, des = surf.detectAndCompute(img, None)
                face_parts_kp.setdefault('entire', []).append(len(kp))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rects = detector(gray, 1)

                for (i, rect) in enumerate(rects):
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    for (name, (j, k)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                        numKp = 0
                        (x, y, w, h) = cv2.boundingRect(np.array([shape[j:k]]))
                        roi = img[y - int(h / 10): y + int(1.1 * h), x - int(w / 10):x + int(1.1 * w)]
                        if roi.shape[0] == 0 or roi.shape[1] == 0:
                            continue
                        for ak in kp:
                            if ((y - h / 10) <= ak.pt[1] <= (y + 1.1 * h)) and ((x - w / 10) <= ak.pt[0] <= (x + 1.1 * w)):
                                numKp += 1
                        face_parts_kp.setdefault(name, []).append(numKp)
    return face_parts_kp


def orbSta(imgDir, imgFormat):
    face_parts_kp = {}
    for subdir, dirs, files in os.walk(imgDir):
        for file in files:
            if file[-4:] == imgFormat:

                img = cv2.imread(subdir + '\\' + file)
                orb = cv2.ORB_create()
                kp, des = orb.detectAndCompute(img, None)
                face_parts_kp.setdefault('entire', []).append(len(kp))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rects = detector(gray, 1)

                for (i, rect) in enumerate(rects):
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    for (name, (j, k)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                        numKp = 0
                        (x, y, w, h) = cv2.boundingRect(np.array([shape[j:k]]))
                        roi = img[y - int(h / 10): y + int(1.1 * h), x - int(w / 10):x + int(1.1 * w)]
                        if roi.shape[0] == 0 or roi.shape[1] == 0:
                            continue
                        for ak in kp:
                            if ((y - h / 10) <= ak.pt[1] <= (y + 1.1 * h)) and ((x - w / 10) <= ak.pt[0] <= (x + 1.1 * w)):
                                numKp += 1
                        face_parts_kp.setdefault(name, []).append(numKp)
    return face_parts_kp


def fastSta(imgDir, imgFormat):
    face_parts_kp = {}
    for subdir, dirs, files in os.walk(imgDir):
        for file in files:
            if file[-4:] == imgFormat:

                img = cv2.imread(subdir + '\\' + file)
                fast = cv2.cv2.FastFeatureDetector_create()
                kp = fast.detect(img, None)
                brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
                kp, des = brief.compute(img, kp)
                face_parts_kp.setdefault('entire', []).append(len(kp))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rects = detector(gray, 1)

                for (i, rect) in enumerate(rects):
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    for (name, (j, k)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                        numKp = 0
                        (x, y, w, h) = cv2.boundingRect(np.array([shape[j:k]]))
                        roi = img[y - int(h / 10): y + int(1.1 * h), x - int(w / 10):x + int(1.1 * w)]
                        if roi.shape[0] == 0 or roi.shape[1] == 0:
                            continue
                        for ak in kp:
                            if ((y - h / 10) <= ak.pt[1] <= (y + 1.1 * h)) and ((x - w / 10) <= ak.pt[0] <= (x + 1.1 * w)):
                                numKp += 1
                        face_parts_kp.setdefault(name, []).append(numKp)
    return face_parts_kp


def akazeSta(imgDir, imgFormat):
    face_parts_kp = {}
    for subdir, dirs, files in os.walk(imgDir):
        for file in files:
            if file[-4:] == imgFormat:

                img = cv2.imread(subdir + '\\' + file)
                akaze = cv2.AKAZE_create()
                kp, des = akaze.detectAndCompute(img, None)
                face_parts_kp.setdefault('entire', []).append(len(kp))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rects = detector(gray, 1)

                for (i, rect) in enumerate(rects):
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    for (name, (j, k)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                        numKp = 0
                        (x, y, w, h) = cv2.boundingRect(np.array([shape[j:k]]))
                        roi = img[y - int(h / 10): y + int(1.1 * h), x - int(w / 10):x + int(1.1 * w)]
                        if roi.shape[0] == 0 or roi.shape[1] == 0:
                            continue
                        for ak in kp:
                            if ((y - h / 10) <= ak.pt[1] <= (y + 1.1 * h)) and ((x - w / 10) <= ak.pt[0] <= (x + 1.1 * w)):
                                numKp += 1
                        face_parts_kp.setdefault(name, []).append(numKp)
    return face_parts_kp


def fpStatistic(record_dict):
    for k, v in record_dict.items():
        print(k)
        print("mean: ", np.mean(v))


img_format = '.jpg'

statistic_results_dir = './feature_point_statistics_in_facial_region/'
if not os.path.exists(statistic_results_dir):
    os.makedirs(statistic_results_dir)

statistics = statistic_results_dir + args['dataset'] + '.txt'
outputfile = open(statistics, 'a+')
sys.stdout = outputfile
output = sys.stdout


base_dir = faces_path + args['dataset'] + '/train/'
real_dir = base_dir + 'real/'
fake_dir = base_dir + 'fake/'
print(args['dataset'] + ':')

sift_real = siftSta(real_dir, img_format)
print('sift_real')
fpStatistic(sift_real)
surf_real = surfSta(real_dir, img_format)
print('surf_real')
fpStatistic(surf_real)
orb_real = orbSta(real_dir, img_format)
print('orb_real')
fpStatistic(orb_real)
fast_real = fastSta(real_dir, img_format)
print('fast_real')
fpStatistic(fast_real)
akaze_real = akazeSta(real_dir, img_format)
print('akaze_real')
fpStatistic(akaze_real)
print('\n')

print(args['dataset'] + ':')

sift_fake = siftSta(fake_dir, img_format)
print('sift_fake')
fpStatistic(sift_fake)
surf_fake = surfSta(fake_dir, img_format)
print('surf_fake')
fpStatistic(surf_fake)
orb_fake = orbSta(fake_dir, img_format)
print('orb_fake')
fpStatistic(orb_fake)
fast_fake = fastSta(fake_dir, img_format)
print('fast_fake')
fpStatistic(fast_fake)
akaze_fake = akazeSta(fake_dir, img_format)
print('akaze_fake')
fpStatistic(akaze_fake)
print('\n')

outputfile.close()
