# -*- coding: utf-8 -*-


import argparse
import warnings
import joblib
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
import os


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="choose datasets: UADFV, DeepfakeTIMIT_HQ, DeepfakeTIMIT_LQ,"
                                                       "Deepfakes_c40, Deepfakes_raw, DFD, DFDC, CelebDF_V2")
ap.add_argument("-f", "--featurePoint", required=True, help="choose feature point algorithm: sift, surf, fast, orb, "
                                                            "akaze")
ap.add_argument("-t", "--FFRFDType", required=True, help="choose type of FFR_FD: ave, no_ave")
args = vars(ap.parse_args())


desLen = {'sift': 128, 'surf': 64, 'orb': 32, 'fast': 32, 'akaze': 61}


model = './models/' + args['FFRFDType'] + '_' + args['dataset'] + '_fast_500'
clf = joblib.load(model)
importances = clf.feature_importances_
barlist = plt.bar(range(256), importances)

plt.xlabel('FFR_FD constructed by ' + args['featurePoint'], fontsize=12)
plt.ylabel("feature importance", fontsize=12)
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
for i in range(desLength, 2 * desLength):
    barlist[i].set_color('#FF0000')
for i in range(2 * desLength, 3 * desLength):
    barlist[i].set_color('#FAA460')
for i in range(3 * desLength, 4 * desLength):
    barlist[i].set_color('#FFFF00')
for i in range(4 * desLength, 5 * desLength):
    barlist[i].set_color('#90EE90')
for i in range(5 * desLength, 6 * desLength):
    barlist[i].set_color('#32CD32')
for i in range(6 * desLength, 7 * desLength):
    barlist[i].set_color('#00BFFF')
for i in range(7 * desLength, 8 * desLength):
    barlist[i].set_color('#4682B4')

plt.xticks(ticks=ticks)
# plt.xticks(ticks=[(i+i-1)*0.5 for i in ticks], labels=['e', 'm', 'im', 'reb', 'leb', 're', 'le', 'n'])
plt.title(args['dataset'], fontsize=15)
# plt.figure()

save_path = './feature importance/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
plt.savefig(
    fname=save_path + args['dataset'] + '_' + args['featurePoint'] + '_.png',  bbox_inches='tight')
# plt.show()
plt.cla()
plt.clf()
plt.close()

