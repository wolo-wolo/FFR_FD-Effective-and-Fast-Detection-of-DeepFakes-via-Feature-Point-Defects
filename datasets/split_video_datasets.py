# -*- coding: UTF-8 -*-


import os, sys
from filePath import *
import argparse
import shutil
sys.path.append('../')


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="choose datasets: UADFV, DeepfakeTIMIT_HQ, DeepfakeTIMIT_LQ,"
                                                       "Deepfakes_c40, Deepfakes_raw, DFD, DFDC, CelebDF_V2")
args = vars(ap.parse_args())


def split_DeepfakeTIMIT_HQ(oriPath):
    realVideos = 'deepfaketimit_real/'
    fakeVideos = 'higher_quality_fake/'

    i = 0
    for num in range(int(0.8 * len(os.listdir(oriPath + realVideos)))):
        for subdir, dirs, files in os.walk(oriPath + realVideos + os.listdir(oriPath + realVideos)[num]):
            for file in files:
                if file[-4:] == '.jpg':
                    i += 1
                    src = os.path.join(subdir, file)
                    dfname = str(i) + '.jpg'
                    dst = os.path.join(train_real_dir, dfname)
                    shutil.copyfile(src, dst)
    for num in range(int(0.8 * len(os.listdir(oriPath + realVideos))), len(os.listdir(oriPath + realVideos))):
        for subdir, dirs, files in os.walk(oriPath + realVideos + os.listdir(oriPath + realVideos)[num]):
            for file in files:
                if file[-4:] == '.jpg':
                    i += 1
                    src = os.path.join(subdir, file)
                    dfname = str(i) + '.jpg'
                    dst = os.path.join(test_real_dir, dfname)
                    shutil.copyfile(src, dst)

    i = 0
    for num in range(int(0.8 * len(os.listdir(oriPath + fakeVideos)))):
        for subdir, dirs, files in os.walk(oriPath + fakeVideos + os.listdir(oriPath + fakeVideos)[num]):
            for file in files:
                if file[-4:] == '.jpg':
                    i += 1
                    src = os.path.join(subdir, file)
                    dfname = str(i) + '.jpg'
                    dst = os.path.join(train_fake_dir, dfname)
                    shutil.copyfile(src, dst)
    for num in range(int(0.8 * len(os.listdir(oriPath + fakeVideos))), len(os.listdir(oriPath + fakeVideos))):
        for subdir, dirs, files in os.walk(oriPath + fakeVideos + os.listdir(oriPath + fakeVideos)[num]):
            for file in files:
                if file[-4:] == '.jpg':
                    i += 1
                    src = os.path.join(subdir, file)
                    dfname = str(i) + '.jpg'
                    dst = os.path.join(test_fake_dir, dfname)
                    shutil.copyfile(src, dst)


def split_DeepfakeTIMIT_LQ(oriPath):
    realVideos = 'deepfaketimit_real/'
    fakeVideos = 'lower_quality_fake/'

    i = 0
    for num in range(int(0.8 * len(os.listdir(oriPath + realVideos)))):
        for subdir, dirs, files in os.walk(oriPath + realVideos + os.listdir(oriPath + realVideos)[num]):
            for file in files:
                if file[-4:] == '.jpg':
                    i += 1
                    src = os.path.join(subdir, file)
                    dfname = str(i) + '.jpg'
                    dst = os.path.join(train_real_dir, dfname)
                    shutil.copyfile(src, dst)
    for num in range(int(0.8 * len(os.listdir(oriPath + realVideos))), len(os.listdir(oriPath + realVideos))):
        for subdir, dirs, files in os.walk(oriPath + realVideos + os.listdir(oriPath + realVideos)[num]):
            for file in files:
                if file[-4:] == '.jpg':
                    i += 1
                    src = os.path.join(subdir, file)
                    dfname = str(i) + '.jpg'
                    dst = os.path.join(test_real_dir, dfname)
                    shutil.copyfile(src, dst)

    i = 0
    for num in range(int(0.8 * len(os.listdir(oriPath + fakeVideos)))):
        for subdir, dirs, files in os.walk(oriPath + fakeVideos + os.listdir(oriPath + fakeVideos)[num]):
            for file in files:
                if file[-4:] == '.jpg':
                    i += 1
                    src = os.path.join(subdir, file)
                    dfname = str(i) + '.jpg'
                    dst = os.path.join(train_fake_dir, dfname)
                    shutil.copyfile(src, dst)
    for num in range(int(0.8 * len(os.listdir(oriPath + fakeVideos))), len(os.listdir(oriPath + fakeVideos))):
        for subdir, dirs, files in os.walk(oriPath + fakeVideos + os.listdir(oriPath + fakeVideos)[num]):
            for file in files:
                if file[-4:] == '.jpg':
                    i += 1
                    src = os.path.join(subdir, file)
                    dfname = str(i) + '.jpg'
                    dst = os.path.join(test_fake_dir, dfname)
                    shutil.copyfile(src, dst)


def split_UADFV(oriPath):
    realVideos = 'real/'
    fakeVideos = 'fake/'

    i = 0
    for subdir, dirs, files in os.walk(oriPath + realVideos):
        for file in files:
            if file[-4:] == '.mp4' and i < 0.85 * len(files):
                i += 1
                src = os.path.join(subdir, file)
                dfname = str(i) + '.mp4'
                dst = os.path.join(train_real_dir, dfname)
                shutil.copyfile(src, dst)
            elif file[-4:] == '.mp4' and i < 1 * len(files):
                i += 1
                src = os.path.join(subdir, file)
                dfname = str(i) + '.mp4'
                dst = os.path.join(test_real_dir, dfname)
                shutil.copyfile(src, dst)

    i = 0
    for subdir, dirs, files in os.walk(oriPath + fakeVideos):
        for file in files:
            if file[-4:] == '.mp4' and i < 0.85 * len(files):
                i += 1
                src = os.path.join(subdir, file)
                dfname = str(i) + '.mp4'
                dst = os.path.join(train_fake_dir, dfname)
                shutil.copyfile(src, dst)
            elif file[-4:] == '.mp4' and i < 1 * len(files):
                i += 1
                src = os.path.join(subdir, file)
                dfname = str(i) + '.mp4'
                dst = os.path.join(test_fake_dir, dfname)
                shutil.copyfile(src, dst)


def split_Deepfakes_c40(oriPath):
    realVideos = 'original_sequences/youtube/c40/videos/'
    fakeVideos = 'manipulated_sequences/Deepfakes/c40/videos/'

    i = 0
    for subdir, dirs, files in os.walk(oriPath + realVideos):
        for file in files:
            if file[-4:] == '.mp4' and i < 0.8 * len(files):
                i += 1
                src = os.path.join(subdir, file)
                dfname = str(i) + '.mp4'
                dst = os.path.join(train_real_dir, dfname)
                shutil.copyfile(src, dst)
            elif file[-4:] == '.mp4' and i < 1 * len(files):
                i += 1
                src = os.path.join(subdir, file)
                dfname = str(i) + '.mp4'
                dst = os.path.join(test_real_dir, dfname)
                shutil.copyfile(src, dst)

    i = 0
    for subdir, dirs, files in os.walk(oriPath + fakeVideos):
        for file in files:
            if file[-4:] == '.mp4' and i < 0.8 * len(files):
                i += 1
                src = os.path.join(subdir, file)
                dfname = str(i) + '.mp4'
                dst = os.path.join(train_fake_dir, dfname)
                shutil.copyfile(src, dst)
            elif file[-4:] == '.mp4' and i < 1 * len(files):
                i += 1
                src = os.path.join(subdir, file)
                dfname = str(i) + '.mp4'
                dst = os.path.join(test_fake_dir, dfname)
                shutil.copyfile(src, dst)


def split_Deepfakes_raw(oriPath):
    realVideos = 'original_sequences/youtube/raw/videos/'
    fakeVideos = 'manipulated_sequences/Deepfakes/raw/videos/'

    i = 0
    for subdir, dirs, files in os.walk(oriPath + realVideos):
        for file in files:
            if file[-4:] == '.mp4' and i < 0.8 * len(files):
                i += 1
                src = os.path.join(subdir, file)
                dfname = str(i) + '.mp4'
                dst = os.path.join(train_real_dir, dfname)
                shutil.copyfile(src, dst)
            elif file[-4:] == '.mp4' and i <= 1 * len(files):
                i += 1
                src = os.path.join(subdir, file)
                dfname = str(i) + '.mp4'
                dst = os.path.join(test_real_dir, dfname)
                shutil.copyfile(src, dst)

    i = 0
    for subdir, dirs, files in os.walk(oriPath + fakeVideos):
        for file in files:
            if file[-4:] == '.mp4' and i < 0.8 * len(files):
                i += 1
                src = os.path.join(subdir, file)
                dfname = str(i) + '.mp4'
                dst = os.path.join(train_fake_dir, dfname)
                shutil.copyfile(src, dst)
            elif file[-4:] == '.mp4' and i <= 1 * len(files):
                i += 1
                src = os.path.join(subdir, file)
                dfname = str(i) + '.mp4'
                dst = os.path.join(test_fake_dir, dfname)
                shutil.copyfile(src, dst)


def split_DFD(oriPath):
    realVideos = 'original_sequences/actors/raw/videos'
    fakeVideos = 'manipulated_sequences/DeepFakeDetection/raw/videos/'

    i = 0
    for subdir, dirs, files in os.walk(oriPath + realVideos):
        for file in files:
            if file[-4:] == '.mp4' and i < 0.8 * len(files):
                i += 1
                src = os.path.join(subdir, file)
                dfname = str(i) + '.mp4'
                dst = os.path.join(train_real_dir, dfname)
                shutil.move(src, dst)
            elif file[-4:] == '.mp4' and i < 1 * len(files):
                i += 1
                src = os.path.join(subdir, file)
                dfname = str(i) + '.mp4'
                dst = os.path.join(test_real_dir, dfname)
                shutil.move(src, dst)

    i = 0
    for subdir, dirs, files in os.walk(oriPath + fakeVideos):
        for file in files:
            if file[-4:] == '.mp4' and i < 0.8 * len(files):
                i += 1
                src = os.path.join(subdir, file)
                dfname = str(i) + '.mp4'
                dst = os.path.join(train_fake_dir, dfname)
                shutil.move(src, dst)
            elif file[-4:] == '.mp4' and i < 1 * len(files):
                i += 1
                src = os.path.join(subdir, file)
                dfname = str(i) + '.mp4'
                dst = os.path.join(test_fake_dir, dfname)
                shutil.move(src, dst)


def split_DFDC(oriPath):
    realVideos = 'real/'
    fakeVideos = 'fake/'

    i = 0
    for subdir, dirs, files in os.walk(oriPath + realVideos):
        for file in files:
            if file[-4:] == '.mp4' and i < 0.8 * len(files):
                i += 1
                src = os.path.join(subdir, file)
                dfname = str(i) + '.mp4'
                dst = os.path.join(train_real_dir, dfname)
                shutil.copyfile(src, dst)
            elif file[-4:] == '.mp4' and i < 1 * len(files):
                i += 1
                src = os.path.join(subdir, file)
                dfname = str(i) + '.mp4'
                dst = os.path.join(test_real_dir, dfname)
                shutil.copyfile(src, dst)

    i = 0
    for subdir, dirs, files in os.walk(oriPath + fakeVideos):
        for file in files:
            if file[-4:] == '.mp4' and i < 0.8 * len(files):
                i += 1
                src = os.path.join(subdir, file)
                dfname = str(i) + '.mp4'
                dst = os.path.join(train_fake_dir, dfname)
                shutil.copyfile(src, dst)
            elif file[-4:] == '.mp4' and i < 1 * len(files):
                i += 1
                src = os.path.join(subdir, file)
                dfname = str(i) + '.mp4'
                dst = os.path.join(test_fake_dir, dfname)
                shutil.copyfile(src, dst)


def split_CelebDF_V2(oriPath):
    path = ['Celeb-real/',
            'YouTube-real/',
            'Celeb-synthesis/']
    labels = [0, 0, 1]
    f = open(oriPath + 'List_of_testing_videos.txt', 'r')  # official label for testing set
    content = f.readlines()
    test_videos = []
    for name in content:
        test_videos.append(name[2:].replace('\n', ''))  # remove '\n' in the official label

    for z in range(len(path)):
        print('process in :', path[z])
        if labels[z] == 0:
            test_num = 0
            for subdir, dirs, files in os.walk(oriPath + path[z]):
                for file in files:
                    video = path[z] + file
                    if file[-4:] == '.mp4' and video in test_videos:
                        test_num += 1
                        src = os.path.join(subdir, file)
                        dfname = file
                        dst = os.path.join(test_real_dir, dfname)
                        shutil.copyfile(src, dst)

            i = 0
            for subdir, dirs, files in os.walk(oriPath + path[z]):
                for file in files:
                    video = path[z] + file
                    if file[-4:] == '.mp4' and (video not in test_videos) and i < 0.8 * (len(files) - test_num):
                        i += 1
                        src = os.path.join(subdir, file)
                        dfname = file
                        dst = os.path.join(train_real_dir, dfname)
                        shutil.copyfile(src, dst)

                    elif file[-4:] == '.mp4' and (video not in test_videos) and i < (len(files) - test_num):
                        i += 1
                        src = os.path.join(subdir, file)
                        dfname = file
                        dst = os.path.join(validation_real_dir, dfname)
                        shutil.copyfile(src, dst)

        if labels[z] == 1:
            test_num = 0
            for subdir, dirs, files in os.walk(oriPath + path[z]):
                for file in files:
                    video = path[z] + file
                    if file[-4:] == '.mp4' and video in test_videos:
                        test_num += 1
                        src = os.path.join(subdir, file)
                        dfname = file
                        dst = os.path.join(test_fake_dir, dfname)
                        shutil.copyfile(src, dst)

            i = 0
            for subdir, dirs, files in os.walk(oriPath + path[z]):
                for file in files:
                    video = path[z] + file
                    if file[-4:] == '.mp4' and (video not in test_videos) and i < 0.8 * (len(files) - test_num):
                        i += 1
                        src = os.path.join(subdir, file)
                        dfname = file
                        dst = os.path.join(train_fake_dir, dfname)
                        shutil.copyfile(src, dst)

                    elif file[-4:] == '.mp4' and (file not in test_videos) and i < (len(files) - test_num):
                        i += 1
                        src = os.path.join(subdir, file)
                        dfname = file
                        dst = os.path.join(validation_fake_dir, dfname)
                        shutil.copyfile(src, dst)


dataPath = original_videos_path[args["dataset"]]
splitPath = split_path[args["dataset"]]
if not os.path.exists(splitPath):
    os.makedirs(splitPath)
train_dir = train_dir[args["dataset"]]
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
test_dir = test_dir[args["dataset"]]
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

train_real_dir = train_real_dir[args["dataset"]]
if not os.path.exists(train_real_dir):
    os.mkdir(train_real_dir)
train_fake_dir = train_fake_dir[args["dataset"]]
if not os.path.exists(train_fake_dir):
    os.mkdir(train_fake_dir)
test_real_dir = test_real_dir[args["dataset"]]
if not os.path.exists(test_real_dir):
    os.mkdir(test_real_dir)
test_fake_dir = test_fake_dir[args["dataset"]]
if not os.path.exists(test_fake_dir):
    os.mkdir(test_fake_dir)

if args["dataset"] == 'DeepfakeTIMIT_HQ':
    split_DeepfakeTIMIT_HQ(dataPath)
if args["dataset"] == 'DeepfakeTIMIT_LQ':
    split_DeepfakeTIMIT_LQ(dataPath)
if args["dataset"] == 'UADFV':
    split_UADFV(dataPath)
if args["dataset"] == 'FF++_Deepfakes_c40':
    split_Deepfakes_c40(dataPath)
if args["dataset"] == 'FF++_Deepfakes_raw':
    split_Deepfakes_raw(dataPath)
if args["dataset"] == 'DFD':
    split_DFD(dataPath)
if args["dataset"] == 'DFDC':
    split_DFDC(dataPath)
if args["dataset"] == 'CelebDF_V2':
    split_CelebDF_V2(dataPath)
