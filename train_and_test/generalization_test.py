# -*- coding: utf-8 -*-


from sklearn import metrics
import numpy as np
import argparse
import joblib
import sys, os


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="choose datasets: UADFV, DeepfakeTIMIT_HQ, DeepfakeTIMIT_LQ,"
                     "FF++_Deepfakes_c40, FF++_Deepfakes_raw, DFD, DFDC, CelebDF_V2")
ap.add_argument("-f", "--featurePoint", required=True, help="choose feature point algorithm: sift, surf, fast, orb, "
                                                            "akaze")
ap.add_argument("-j", "--jobs", type=int, default=-1, help="choose jobs to run when training")
ap.add_argument("-t", "--FFRFDType", required=True, help="choose type of FFR_FD: ave, no_ave")
args = vars(ap.parse_args())


desLen = {'sift': 128, 'surf': 64, 'orb': 32, 'fast': 32, 'akaze': 61}

output = sys.stdout
output_file = open(args['FFRFDType'] + '_generalization_test_results.txt', 'a+')
sys.stdout = output_file

trainingSet = 'DeepfakeTIMIT_HQ'

print('training set:', trainingSet)
model = './models/' + args['FFRFDType'] + '_' + trainingSet + '_fast_200'
clf = joblib.load(model)

for testSet in ['DeepfakeTIMIT_LQ', 'UADFV',
    'FF++_Deepfakes_raw', 'FF++_Deepfakes_c40',
    'DFD', 'DFDC', 'CelebDF_V2']:
    print('test Set:', testSet)

    realTest = '../FFR_FD_' + args['FFRFDType'] + '/test/' + \
               testSet + '_real_fast_FFR_FD.npy'
    fakeTest = '../FFR_FD_' + args['FFRFDType'] + '/test/' + \
               testSet + '_fake_fast_FFR_FD.npy'
    realTestDes = np.load(realTest)
    fakeTestDes = np.load(fakeTest)
    realTestLabels = np.zeros(realTestDes.shape[0])
    fakeTestLabels = np.ones(fakeTestDes.shape[0])
    x_test = np.vstack((realTestDes, fakeTestDes))
    y_test = np.concatenate((realTestLabels, fakeTestLabels))

    y_pred = clf.predict_proba(x_test)[:, 1]
    preds = clf.predict(x_test)

    print("AUC Score (Train): %f" % metrics.roc_auc_score(y_test, y_pred))
