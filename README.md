# FFR_FD: Effective and Fast Detection of DeepFakes Based on Feature Point Defects

[Code for paper 'FFR_FD: Effective and Fast Detection of DeepFakes Based on Feature Point Defects']



#   Experiment Environment
Our development env is shown in:
```
requirements.txt
```


# 1) Datasets
Download Original Datasets to the corresponding *./datasets/Videos/* folders.
* [DeepfakeTIMIT_HQ](https://www.idiap.ch/dataset/deepfaketimit) 
* [DeepfakeTIMIT_LQ](https://www.idiap.ch/dataset/deepfaketimit) 
* [UADFV](https://github.com/danmohaha/WIFS2018_In_Ictu_Oculi) 
* [FF++_c40](https://github.com/ondyari/FaceForensics) 
* [FF++_raw](https://github.com/ondyari/FaceForensics) 
* [DFD](https://ai.googleblog.com/2019/09/contributing-data-to-deepfake-detection.html) 
* [DFDC](https://www.kaggle.com/c/deepfake-detection-challenge) 
* [CelebDF_V2](https://github.com/danmohaha/celeb-deepfakeforensics) 

### Split Datasets
Split Videos to the training set and testing set:
```
cd datasets
python split_video_datasets.py -h
```

###extract the frames:
```
python extract_frames.py -h
```

use the S3Fd Detector and Fan Aligner to extract facial images.
[faceswap github].(https://github.com/deepfakes/faceswap) 


#2) Feature Points Statistics
First, download the [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
to the *./code_for_ACM_MM21_paper_1009/*

```
cd ..
python feature_point_statistics.py -h
```


#3) FFR_FD

```
cd 'construct_FFR_FD_for_datasets'
python FFR_FD_no_ave_train_set.py
python FFR_FD_no_ave_test_set.py -h
python FFR_FD_ave_train_set.py -h
python FFR_FD_ave_test_set.py -h 
```


#3) Differences in FFR_FD
```
cd "differences_in_FFR_FD"
python statistics_differences_of_FFR_FD.py -h
```


#4) Train and Test
```
cd train_and_test
python train_and_test.py -h
```

###Generalization Test:
```
python generalization_test.py -h
```

#6)features importance：
```
python features_importances.py -h
```


# FFR_FD-Effective-and-Fast-Detection-of-DeepFakes-Based-on-Feature-Point-Defects
# FFR_FD-Effective-and-Fast-Detection-of-DeepFakes-Based-on-Feature-Point-Defects
