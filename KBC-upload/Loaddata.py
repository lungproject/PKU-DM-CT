import os
# from PIL import Image  
import csv
import numpy as np
import pandas as pd


def load_testdata(rid):  # ct window

    csvPath = "./Data/info.csv"
    df = pd.read_csv(csvPath, sep=',')

    df['patient_id'] = './Data/DataCropVolumez10/' + df['patient_id'] + '.npy'
    df_test = df[df['id_group'] == 2]

    df_test = pd.concat([df_test], ignore_index=True)
    df_test = df_test[["patient_id", "label"]]

    test = np.load(df_test['patient_id'][0])
    test = np.expand_dims(test, 0)

    for patientID in df_test['patient_id']:
        testtemp = np.load(patientID)
        testtemp = np.expand_dims(testtemp, 0)
        test = np.concatenate((test, testtemp), axis=0)

    test = test[1:, :, :, :]
    testlabel = df_test['label']


    test1 = np.asarray(test[:, :, :, 0:3], dtype="float32")
    test2 = np.asarray(test[:, :, :, 3:6], dtype="float32")
    test3 = np.asarray(test[:, :, :, 6:9], dtype="float32")
    test4 = np.asarray(test[:, :, :, 9:12], dtype="float32")
    test5 = np.asarray(test[:, :, :, 12:15], dtype="float32")
    test6 = np.asarray(test[:, :, :, 15:18], dtype="float32")
    test7 = np.asarray(test[:, :, :, 18:21], dtype="float32")
    test8 = np.asarray(test[:, :, :, 21:24], dtype="float32")
    test9 = np.asarray(test[:, :, :, 24:27], dtype="float32")


    testlabel = np.asarray(testlabel, dtype="float32")

    return test1, test2, test3, test4, test5, test6, test7, test8, test9, testlabel
