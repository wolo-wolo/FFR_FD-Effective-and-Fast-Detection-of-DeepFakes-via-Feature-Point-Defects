# -*- coding: UTF-8 -*-


import cv2
from filePath import *
import argparse
import os, sys
sys.path.append('../')


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="choose datasets: UADFV, DeepfakeTIMIT_HQ, DeepfakeTIMIT_LQ,"
                                                       "Deepfakes_c40, Deepfakes_raw, DFD, DFDC, CelebDF_V2")
args = vars(ap.parse_args())


print('process start')

framesNum = []
for path in [
    train_real_dir[args["dataset"]],
    train_fake_dir[args["dataset"]],
    test_real_dir[args["dataset"]],
    test_fake_dir[args["dataset"]]
]:
    frames = 0
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file[-4:] == '.mp4':
                src_video = os.path.join(subdir, file)
                frame = cv2.VideoCapture(src_video)
                frame_nums = frame.get(7)
                frames += frame_nums
    framesNum.append(frames)
print(framesNum)


i = 0
labels = [0, 0, 1, 1]
for path in [
    train_real_dir[args["dataset"]],
    train_fake_dir[args["dataset"]],
    test_real_dir[args["dataset"]],
    test_fake_dir[args["dataset"]]
]:
    frames = 0

    if labels[i] == 0:
        if args["dataset"] == 'DFDC':
            frameFrequency = round(framesNum[i] / 18000, 0)
        elif args["dataset"] == 'DeepfakeTIMIT_LQ' or 'DeepfakeTIMIT_HQ' or 'UADFV':
            frameFrequency = 1
        else:
            frameFrequency = round(framesNum[i] / 30000, 0)

    elif labels[i] == 1:
        if args["dataset"] == 'DFDC':
            frameFrequency = round(framesNum[i] / 4500, 0)
        elif args["dataset"] == 'DeepfakeTIMIT_LQ' or 'DeepfakeTIMIT_HQ' or 'UADFV':
            frameFrequency = 1
        else:
            frameFrequency = round(framesNum[i] / 10000, 0)

    i += 1

    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file[-4:] == '.mp4':
                src_video = os.path.join(subdir, file)
                times = 0

                outPutDirName = str(frames_path + subdir[subdir.find(args["dataset"]):])
                if not os.path.exists(outPutDirName):
                    os.makedirs(outPutDirName)

                frame = cv2.VideoCapture(src_video)
                while True:
                    times += 1
                    res, image = frame.read()
                    if not res:
                        break
                    if times % frameFrequency == 0:
                        cv2.imwrite(outPutDirName + '/' + file + '%' + str(times) + '.jpg', image)
                        frames += 1
                frame.release()

    print('%s has extracted frames: %d' % (path, frames))
