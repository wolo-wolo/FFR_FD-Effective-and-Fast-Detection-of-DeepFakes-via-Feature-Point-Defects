# -*- coding: UTF-8 -*-


import os

# The path to the downloaded DeepFake video datasets
original_videos_path = {
    'UADFV': './datasets/Videos/UADFV/',
    'DeepfakeTIMIT_HQ': './datasets/Videos/DeepfakeTIMIT_frames/',
    'DeepfakeTIMIT_LQ': './datasets/Videos/DeepfakeTIMIT_frames/',  # provided frames
    'FF++_Deepfakes_c40': './datasets/Videos/FF++/Deepfakes/c40/',
    'FF++_Deepfakes_raw': './datasets/Videos/FF++/Youtube/raw/',
    'DFD': './datasets/Videos/DFD/',
    'DFDC': './datasets/Videos/DFDC/',
    'CelebDF_V2': './datasets/Videos/CelebDF_V2/',
}


# The path of the videos divided into the training set and the test set.
split_path = {
    'UADFV': './datasets/video_datasets/UADFV/',
    'DeepfakeTIMIT': './datasets/video_datasets/DeepfakeTIMIT_frames/',
    'DeepfakeTIMIT_LQ': './datasets/video_datasets/DeepfakeTIMIT_LQ_frames/',
    'FF++_Deepfakes': './datasets/video_datasets/FF++_Deepfakes/',
    'FF++_Deepfakes_raw': './datasets/video_datasets/FF++_Deepfakes_raw/',
    'FF++_Face2Face': './datasets/video_datasets/FF++_Face2Face/',
    'FF++_FaceSwap': './datasets/video_datasets/FF++_FaceSwap/',
    'FF++_NeuralTextures': './datasets/video_datasets/FF++_NeuralTextures/',
    'DFD': './datasets/video_datasets/DFD/raw/',
    'DFDC': './datasets/video_datasets/DFDC/',
    'CelebDF_V2': './datasets/video_datasets/CelebDF_V2/',
}

train_dir = {}
test_dir = {}
for f in split_path.keys():
    train_dir[f] = os.path.join(split_path[f], 'train')
    test_dir[f] = os.path.join(split_path[f], 'test')

train_real_dir = {}
train_fake_dir = {}
validation_real_dir = {}
validation_fake_dir = {}
test_real_dir = {}
test_fake_dir = {}

for f in split_path.keys():
    train_real_dir[f] = os.path.join(train_dir[f], 'real')
    train_fake_dir[f] = os.path.join(train_dir[f], 'fake')
    test_real_dir[f] = os.path.join(test_dir[f], 'real')
    test_fake_dir[f] = os.path.join(test_dir[f], 'fake')


# The path to the extracted frame
frames_path = './datasets/frames_datasets/'


# The path to the extracted faces
faces_path = './datasets/faces/'
