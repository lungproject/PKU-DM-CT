from __future__ import print_function

import logging
import os

import pandas as pd

import numpy as np
import math

if __name__ == '__main__':
    RS = np.load('./results/predictRS.npy')
    DLS = np.load('./results/predictdls.npy')
    DLS = DLS[:,1]
    glmcoef = [6.410277428,6.10950259,-4.910803672]
    num = np.shape(RS)[0]
    b = np.ones((1, num))
    feature = np.vstack((DLS, RS,b))
    DRS = np.dot(glmcoef, feature)
    # obtain methyinfo
    methyinfo = pd.read_csv('./data/methylsmall.txt', sep='\s+')
    methyvalue = methyinfo.values
    methyvaluedf=pd.DataFrame(methyvalue[:,1:],index=methyvalue[:,0])
    featurename = ['Block7087','Block1926','Block1674','Block4605','Block19213']
    methycoef = [7.905325878,7.66336516,6.638233344,0.175756046,55.73712453,-1.990545058]
    methyfeature = methyvaluedf.loc[featurename[0]]
    methyfeature = np.asfarray(methyfeature, dtype=float)

    for nameid in range(1, 5):
        tempfeature = methyvaluedf.loc[featurename[nameid]]
        tempfeature = np.asfarray(tempfeature, dtype=float)
        methyfeature = np.vstack((methyfeature,tempfeature))

    num = np.shape(methyfeature)[1]
    b = np.ones((1, num))
    methyfeature = np.vstack((methyfeature, b))
    methyrs = np.dot(methycoef, methyfeature)

    finalcoef = [0.903569740398742, 1.279819772110745, -1.812940181289075]

    allfeature = np.vstack((DRS,methyrs))
    num = np.shape(allfeature)[1]
    b = np.ones((1, num))
    allfeature = np.vstack((allfeature, b))
    finalpred = np.dot(finalcoef, allfeature)

    score = math.exp(finalpred[0]) / (math.exp(finalpred[0]) + 1)
    for id in range(1,num):
        temps = math.exp(finalpred[id])/(math.exp(finalpred[id])+1)
        score = np.vstack((score,temps))
    score[DRS>=2.5712] = 1
    score[DRS<-0.08166] = 0
    print(score)
