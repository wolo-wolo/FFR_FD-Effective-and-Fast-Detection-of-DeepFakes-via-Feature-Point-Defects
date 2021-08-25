# -*- coding: UTF-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import argparse
import warnings
import os
from scipy import stats
warnings.filterwarnings('ignore')


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="choose datasets: UADFV, DeepfakeTIMIT_HQ, DeepfakeTIMIT_LQ,"
                     "FF++_Deepfakes_c40, FF++_Deepfakes_raw, DFD, DFDC, CelebDF_V2")
ap.add_argument("-f", "--featurePoint", required=True, help="choose feature point algorithm: sift, surf, fast, orb, "
                                                            "akaze")
ap.add_argument("-t", "--FFRFDType", required=True, help="choose type of FFR_FD: ave, no_ave")
args = vars(ap.parse_args())


rng = np.random.RandomState(2020)

realF = '../FFR_FD_' + args['FFRFDType'] + '/train/' + \
                args['dataset'] + '_real_' + args['featurePoint'] + '_FFR_FD.npy'
fakeF = '../FFR_FD_' + args['FFRFDType'] + '/train/' + \
                args['dataset'] + '_fake_' + args['featurePoint'] + '_FFR_FD.npy'
realFeatures = np.load(realF)
fakeFeatures = np.load(fakeF)

realDesSta = {}
fakeDesSta = {}
for key, value in stats.describe(realFeatures)._asdict().items():
    realDesSta[key] = value
for key, value in stats.describe(fakeFeatures)._asdict().items():
    fakeDesSta[key] = value

for key, value in stats.describe(fakeFeatures)._asdict().items():
    print(key)
    if (key == 'nobs') or (key == 'minmax'):
        continue

    plt.rcParams["axes.unicode_minus"] = False
    sta_real = realDesSta[key]
    sta_fake = fakeDesSta[key]
    sta_diff = realDesSta[key] - fakeDesSta[key]

    x = list(range(len(sta_real)))
    print(len(x))

    barlist = plt.bar(x, sta_diff, fc="r")

    plt.xlabel('dimension of FFR_FD', fontsize=12)
    plt.ylabel("difference value of " + key, fontsize=12)
    if args['featurePoint'] == 'sift':
        desLength = 128
        ticks = [0, 128, 256, 384, 512, 640, 768, 896, 1024]
    elif args['featurePoint'] == 'surf':
        desLength = 64
        ticks = [0, 64, 128, 192, 256, 320, 384, 448, 512]
    elif args['featurePoint'] == 'fast' or 'orb':
        desLength = 32
        ticks = [0, 32, 64, 96, 128, 160, 192, 224, 256]
    elif args['featurePoint'] == 'akaze':
        desLength = 61
        ticks = [0, 61, 122, 183, 244, 305, 366, 427, 488]

    for i in range(desLength):
        barlist[i].set_color('#A52A2A')
    for i in range(desLength, 2*desLength):
        barlist[i].set_color('#FF0000')
    for i in range(2*desLength, 3*desLength):
        barlist[i].set_color('#FAA460')
    for i in range(3*desLength, 4*desLength):
        barlist[i].set_color('#FFFF00')
    for i in range(4*desLength, 5*desLength):
        barlist[i].set_color('#90EE90')
    for i in range(5*desLength, 6*desLength):
        barlist[i].set_color('#32CD32')
    for i in range(6*desLength, 7*desLength):
        barlist[i].set_color('#00BFFF')
    for i in range(7*desLength, 8*desLength):
        barlist[i].set_color('#4682B4')

    plt.xticks(ticks=ticks)
    # plt.xticks(ticks=[(i+i-1)*0.5 for i in ticks], labels=['e', 'm', 'im', 'reb', 'leb', 're', 'le', 'n'])
    plt.title(args['featurePoint'], fontsize=15)

    save_path = './FFR_FD_differences_between_real_fake/' + args['dataset'] + '/' + args['FFRFDType'] + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(
        fname=save_path + args['dataset'] + '_' + args['featurePoint'] + '_' + key + '_.png',  bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()
