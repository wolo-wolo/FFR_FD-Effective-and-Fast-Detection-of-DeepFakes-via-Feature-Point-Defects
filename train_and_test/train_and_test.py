# -*- coding: utf-8 -*-


import sys, os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import argparse
import joblib


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
result_dir = './Test_Results/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
output_file = open(result_dir + args['dataset'] + '_classification_auc_FFD_FD_' + args['FFRFDType'] + '.txt', 'a+')
sys.stdout = output_file

print('FFR_FD_from_' + args['featurePoint'] + ':')

for subtrees in [200, 500, 800]:
    print('subtrees:', subtrees)

    realTrain = '../FFR_FD_' + args['FFRFDType'] + '/train/' + \
                args['dataset'] + '_real_' + args['featurePoint'] + '_FFR_FD.npy'
    fakeTrain = '../FFR_FD_' + args['FFRFDType'] + '/train/' + \
                args['dataset'] + '_fake_' + args['featurePoint'] + '_FFR_FD.npy'
    realTrainDes = np.load(realTrain)
    fakeTrainDes = np.load(fakeTrain)
    realTrainLabels = np.zeros(realTrainDes.shape[0])
    fakeTrainLabels = np.ones(fakeTrainDes.shape[0])

    x_train = np.vstack((realTrainDes, fakeTrainDes))
    y_train = np.concatenate((realTrainLabels, fakeTrainLabels))

    realTest = '../FFR_FD_' + args['FFRFDType'] + '/test/' + \
                args['dataset'] + '_real_' + args['featurePoint'] + '_FFR_FD.npy'
    fakeTest = '../FFR_FD_' + args['FFRFDType'] + '/test/' + \
                args['dataset'] + '_fake_' + args['featurePoint'] + '_FFR_FD.npy'
    realTestDes = np.load(realTest)
    fakeTestDes = np.load(fakeTest)
    realTestLabels = np.zeros(realTestDes.shape[0])
    fakeTestLabels = np.ones(fakeTestDes.shape[0])

    x_test = np.vstack((realTestDes, fakeTestDes))
    y_test = np.concatenate((realTestLabels, fakeTestLabels))

    rf = RandomForestClassifier(n_estimators=subtrees, random_state=12, n_jobs=-1)
    # start_time = time.time()
    rf.fit(x_train, y_train)
    # end_time = time.time()
    # times = (end_time - start_time)
    # print('time:', times)

    y_pred = rf.predict_proba(x_test)[:, 1]
    preds = rf.predict(x_test)

    print("Test AUC Score: %f" % metrics.roc_auc_score(y_test, y_pred))

    if subtrees == 200 or subtrees == 500:
        model_dir = './models/'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        save_model = model_dir + args['FFRFDType'] + '_' + args['dataset'] + '_' + args['featurePoint'] + '_' + str(subtrees)
        joblib.dump(rf, save_model)
    elif subtrees == 800:
        break
