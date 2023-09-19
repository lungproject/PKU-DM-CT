from __future__ import print_function
from sklearn import datasets
import datetime
import keras
import numpy as np
import pdb
import random
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL

# from keras.callbacks import CSVLogger
from keras import applications
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
# from keras.applications import ResNet50
import sklearn
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split


# from keras.applications.inception_v3 import InceptionV3
# from keras.models import  Sequential
from keras.layers import PReLU,Input,Dense, Dropout, Flatten, Activation,AveragePooling2D, GlobalAveragePooling2D,BatchNormalization,GlobalMaxPooling2D,UpSampling2D
from keras_cv.layers import DropBlock2D

from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Model
from keras import regularizers
# from keras.layers.advanced_activations import PReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, EarlyStopping,ModelCheckpoint,ReduceLROnPlateau

from tensorflow.keras.optimizers import SGD
from keras.constraints import maxnorm
# from sklearn.cross_validation import StratifiedKFold
from keras.models import load_model
from keras import backend as K
import tensorflow as tf



from Loaddata import load_testdata

# import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#from Loaddata import load_data

# from sklearn.metrics import roc_auc_score
# from scipy.io import loadmat



# from sklearn.cross_validation import train_validate_split
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()



def focal_loss(y_true, y_pred):
   gamma = 2
   alpha = .0001
   pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
   pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
   return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

def auc(y_true, y_pred):
    auc = tf.keras.metrics.AUC(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc
# def auc(y_true, y_pred):
#     auc_value, auc_op = tf.metrics.auc(y_true, y_pred)#[1]
#     K.get_session().run(tf.global_variables_initializer())
#     K.get_session().run(tf.local_variables_initializer())
#     K.get_session().run(auc_op)
#     auc = K.get_session().run(auc_value)

def auc_calculate(labels, preds, n_bins=1000):
    postive_len = sum(labels)
    negative_len = len(labels) - postive_len
    total_case = postive_len * negative_len
    pos_histogram = [0 for _ in range(n_bins)]
    neg_histogram = [0 for _ in range(n_bins)]


    for i in range(len(labels)):
        nth_bin = int(preds[i] * n_bins)
        if labels[i] == 1:
            pos_histogram[nth_bin] += 1
        else:
            neg_histogram[nth_bin] += 1
    accumulated_neg = 0
    satisfied_pair = 0
    for i in range(n_bins):
        satisfied_pair += (pos_histogram[i] * accumulated_neg + pos_histogram[i] * neg_histogram[i] * 0.5)
        accumulated_neg += neg_histogram[i]

    # print(satisfied_pair)
    # print(total_case)
    return satisfied_pair / float(total_case)
#     return auc
def sensitivity(y_true, y_pred):
    # true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    # possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    # return true_positives / (possible_positives + K.epsilon())
    TP = tf.reduce_sum(y_true[:, 1] * tf.round(y_pred[:, 1]))
    TN = tf.reduce_sum((1 - y_true[:, 1]) * (1 - tf.round(y_pred[:, 1])))
    FP = tf.reduce_sum((1 - y_true[:, 1]) * tf.round(y_pred[:, 1]))
    FN = tf.reduce_sum(y_true[:, 1] * (1 - tf.round(y_pred[:, 1])))
    sen=TP/(TP + FN + K.epsilon())
    return sen


def specificity(y_true, y_pred):
    # true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    # possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    # return true_negatives / (possible_negatives + K.epsilon())

    TP = tf.reduce_sum(y_true[:,1] * tf.round(y_pred[:,1]))
    TN = tf.reduce_sum((1 - y_true[:,1]) * (1 - tf.round(y_pred[:,1])))
    FP = tf.reduce_sum((1 - y_true[:,1]) * tf.round(y_pred[:,1]))
    FN = tf.reduce_sum(y_true[:,1] * (1 - tf.round(y_pred[:,1])))
    spec = TN / (TN + FP + K.epsilon())
    return spec


#精确率评价指标
def metric_precision(y_true,y_pred):
    TP=tf.reduce_sum(y_true[:,1] *tf.round(y_pred[:,1]))
    TN=tf.reduce_sum((1-y_true[:,1])*(1-tf.round(y_pred[:,1])))
    FP=tf.reduce_sum((1-y_true[:,1])*tf.round(y_pred[:,1]))
    FN=tf.reduce_sum(y_true[:,1] *(1-tf.round(y_pred[:,1])))
    precision=TP/(TP + FP + K.epsilon())
    return precision

#召回率评价指标
def metric_recall(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    recall=TP/(TP+FN)
    return recall

def metric_F1score(y_true,y_pred):
    TP=tf.reduce_sum(y_true[:,1]*tf.round(y_pred[:,1]))
    TN=tf.reduce_sum((1-y_true[:,1])*(1-tf.round(y_pred[:,1])))
    FP=tf.reduce_sum((1-y_true[:,1])*tf.round(y_pred[:,1]))
    FN=tf.reduce_sum(y_true[:,1]*(1-tf.round(y_pred[:,1])))
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    F1score=2*precision*recall/(precision+recall)
    return F1score
#FUSE CROSS VALIDATION


if __name__ == "__main__":



    model = load_model('./model/KBCdeepmodel.hdf5',custom_objects={'focal_loss':focal_loss,'sensitivity':sensitivity,'specificity':specificity,'metric_F1score':metric_F1score})
    # model.summary()


    rid=140
    num_classes=2

    x_test1, x_test2, x_test3, x_test4, x_test5, x_test6, x_test7, x_test8, x_test9, y_test = load_testdata(rid=rid)

    y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

    predicttest= model.predict( (x_test1,x_test2,x_test3,x_test4,x_test5,x_test6,x_test7,x_test8,x_test9), verbose=1)#ADCdata1,

    pathtest = "./results/predictdls.npy"

    outfile_x = open(pathtest,'wb')
    np.save(outfile_x, predicttest)
    outfile_x.close()





