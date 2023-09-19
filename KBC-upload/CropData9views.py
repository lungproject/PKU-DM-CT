#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
import shutil
import scipy.misc
from scipy.ndimage import zoom
import nrrd
import logging
import torch

def _boundingBox(A):
    B = np.argwhere(A)
    if A.ndim == 3:
        (zstart, ystart, xstart), (zstop, ystop, xstop) = B.min(axis=0), B.max(axis=0) + 1
        return (zstart, ystart, xstart), (zstop, ystop, xstop)
    elif A.ndim == 2:
        (ystart, xstart), (ystop, xstop) = B.min(axis=0), B.max(axis=0) + 1
        return (ystart, xstart), (ystop, xstop)
    else:
        print('box err')
        return


def normalize(image, chestwl=-400.0, chestww=1500.0):
    # https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
    Low = chestwl-0.5*chestww
    High = chestwl+0.5*chestww
    image[image>High] = High
    image[image<Low] = Low
    image = (image - Low) / (High - Low)
    return image


def CropZoom(masks, patches):
    # (ystart, xstart), (ystop, xstop) = _boundingBox(masks)
    # ystart = max(ystart - 8, 0)
    # xstart = max(xstart - 8, 0)
    # ystop = min(ystop + 8, 64)
    # xstop = min(xstop + 8, 64)
    # print(ystop-ystart, xstop-xstart)
    # print((ystart, xstart), (ystop, xstop))
    # patch = patches[ystart:ystop,xstart:xstop]


    arrayUseZoom64 = zoom(patches, (64. / patches.shape[0], 64. / patches.shape[1]), order=3)
    x_tensor = torch.from_numpy(arrayUseZoom64)
    #
    mean = np.mean(arrayUseZoom64)
    std = np.std(arrayUseZoom64)
    #
    x_tensor= (x_tensor - mean) / std
    arrayUseZoom64_ed = x_tensor.numpy()
    # # arrayUseZoom64.shape
    arrayUseZoom64_ed = np.expand_dims(arrayUseZoom64_ed , -1)
    # arrayUseZoom64_ed = np.expand_dims(arrayUseZoom64, -1)
    # arrayUseZoom64_3d = np.concatenate((arrayUseZoom64_ed, arrayUseZoom64_ed, arrayUseZoom64_ed), axis=-1)
    return arrayUseZoom64_ed


def Patch_views(arrayCrop, maskCrop, views=None):
    if views == 1:
        patches = arrayCrop[32, :, :]
        masks = maskCrop[32, :, :]
    elif views == 2:
        patches = arrayCrop[:, 32, :]
        masks = maskCrop[:, 32, :]
    elif views == 3:
        patches = arrayCrop[:, :, 32]
        masks = maskCrop[:, :, 32]
    elif views == 4:
        patches = None
        masks = None
        for i in range(64):
            if patches is None:
                masks = np.expand_dims(maskCrop[:, i, i], 1)
                patches = np.expand_dims(arrayCrop[:, i, i], 1)
            else:
                masks = np.concatenate((masks, np.expand_dims(maskCrop[:, i, i], 1)), axis=1)
                patches = np.concatenate((patches, np.expand_dims(arrayCrop[:, i, i], 1)), axis=1)
    elif views == 5:
        patches = None
        masks = None
        for i in range(64):
            if patches is None:
                masks = np.expand_dims(maskCrop[i, :, i], 1)
                patches = np.expand_dims(arrayCrop[i, :, i], 1)
            else:
                masks = np.concatenate((masks, np.expand_dims(maskCrop[i, :, i], 1)), axis=1)
                patches = np.concatenate((patches, np.expand_dims(arrayCrop[i, :, i], 1)), axis=1)
    elif views == 6:
        patches = None
        masks = None
        for i in range(64):
            if patches is None:
                masks = np.expand_dims(maskCrop[i, i, :], 1)
                patches = np.expand_dims(arrayCrop[i, i, :], 1)
            else:
                masks = np.concatenate((masks, np.expand_dims(maskCrop[i, i, :], 1)), axis=1)
                patches = np.concatenate((patches, np.expand_dims(arrayCrop[i, i, :], 1)), axis=1)
    elif views == 7:
        patches = None
        masks = None
        for i in range(64):
            if patches is None:
                masks = np.expand_dims(maskCrop[:, i, 63 - i], 1)
                patches = np.expand_dims(arrayCrop[:, i, 63 - i], 1)
            else:
                masks = np.concatenate((masks, np.expand_dims(maskCrop[:, i, 63 - i], 1)), axis=1)
                patches = np.concatenate((patches, np.expand_dims(arrayCrop[:, i, 63 - i], 1)), axis=1)
    elif views == 8:
        patches = None
        masks = None
        for i in range(64):
            if patches is None:
                masks = np.expand_dims(maskCrop[i, :, 63 - i], 1)
                patches = np.expand_dims(arrayCrop[i, :, 63 - i], 1)
            else:
                masks = np.concatenate((masks, np.expand_dims(maskCrop[i, :, 63 - i], 1)), axis=1)
                patches = np.concatenate((patches, np.expand_dims(arrayCrop[i, :, 63 - i], 1)), axis=1)
    elif views == 9:
        patches = None
        masks = None
        for i in range(64):
            if patches is None:
                masks = np.expand_dims(maskCrop[i, 63 - i, :], 1)
                patches = np.expand_dims(arrayCrop[i, 63 - i, :], 1)
            else:
                masks = np.concatenate((masks, np.expand_dims(maskCrop[i, 63 - i, :], 1)), axis=1)
                patches = np.concatenate((patches, np.expand_dims(arrayCrop[i, 63 - i, :], 1)), axis=1)
    return masks, patches


def sortKey(elem):
    return int(elem[2:])


def window(img):
    win_min = -400
    win_max = 1500

    for i in range(img.shape[0]):
        img[i] = 255.0 * (img[i] - win_min) / (win_max - win_min)
        min_index = img[i] < 0
        img[i][min_index] = 0
        max_index = img[i] > 255
        img[i][max_index] = 255
        img[i] = img[i] - img[i].min()
        c = float(255) / img[i].max()
        img[i] = img[i] * c

    return img.astype(np.uint8)


if __name__ == "__main__":


    ctDataPath = "./Data/Org/"
    ctSavePath = "./Data/DataCropVolumez10/"

    caseIDs = list(set([i.split(' ')[0] for i in os.listdir(ctDataPath)]))
    # print (caseIDs)
    caseIDs.sort(key=sortKey)
    # caseIDs2 = caseIDs[0:1]
    print(len(caseIDs))
    for patientID in caseIDs:
        print("***   {}   ***".format(patientID))

        ###############   saved patientID
        savePath = ctSavePath
        if not os.path.exists(savePath):
            os.mkdir(savePath)
        ###############   Skip processed files
        #     saveHStest = savePath +  "view9HS.npy"
        #     if os.path.exists(saveHStest):
        #         continue
        ###################################################

        testArray = ctDataPath + patientID + " Image mediastinum.nrrd"
        testLabel = ctDataPath + patientID + " Segmentation.seg.nrrd"
        ## using simpleITK to load and save data.
        itk_mask = sitk.ReadImage(testLabel)
        mask = sitk.GetArrayFromImage(itk_mask)
        ## using simpleITK to load and save data.
        array, optionsimg = nrrd.read(testArray)
        itk_array = sitk.ReadImage(testArray)
        array = sitk.GetArrayFromImage(itk_array)
        # array = window(array)

        print("1: ", array.max(), array.min())
        array = normalize(array)
        print("2: ", array.max(), array.min())

        try:
            assert (mask.shape == array.shape)
        except:
            print("mask shape:", mask.shape)
            print("array shape:", array.shape)
            continue

        ## lesion location
        mask[mask <= 0.5] = 0
        mask[mask > 0.5] = 1
        (zstart, ystart, xstart), (zstop, ystop, xstop) = _boundingBox(mask)

        zsize = zstop - zstart
        ysize = ystop - ystart
        xsize = xstop - xstart

        zcenter = (zstop + zstart) // 2
        ycenter = (ystop + ystart) // 2
        xcenter = (xstop + xstart) // 2

        zz = int(10/optionsimg['space directions'][2, 2] )
        yy = int(10/optionsimg['space directions'][0, 0] )
        xx = int(10/optionsimg['space directions'][1, 1] )

        cropSize = max(zsize, ysize, xsize)
        deltas = max(zz,yy,xx)

        zstartUse = zcenter - cropSize // 2 - deltas
        zstopUse = zcenter + cropSize // 2 + deltas
        ystartUse = ycenter - cropSize // 2 - deltas
        ystopUse = ycenter + cropSize // 2 + deltas
        xstartUse = xcenter - cropSize // 2 - deltas
        xstopUse = xcenter + cropSize // 2 + deltas
        if zstartUse<0:
            deltas = zcenter - cropSize // 2-1
            zstartUse = zcenter - cropSize // 2 - deltas
            zstopUse = zcenter + cropSize // 2 + deltas
            ystartUse = ycenter - cropSize // 2 - deltas
            ystopUse = ycenter + cropSize // 2 + deltas
            xstartUse = xcenter - cropSize // 2 - deltas
            xstopUse = xcenter + cropSize // 2 + deltas

        print(cropSize,deltas,zstartUse,zstopUse)

        maskUse = mask[zstartUse:zstopUse, ystartUse:ystopUse, xstartUse:xstopUse]
        tumorUse = array[zstartUse:zstopUse, ystartUse:ystopUse, xstartUse:xstopUse]
       
        [depth, height, width] = tumorUse.shape
        print(depth, height, width)
        arrayCrop = zoom(tumorUse, (64 * 1.0 / depth, 64 * 1.0 / height, 64 * 1.0 / width), order=3)
        maskCrop = zoom(maskUse, (64 * 1.0 / depth, 64 * 1.0 / height, 64 * 1.0 / width), order=3)

        maskCrop[maskCrop <= 0.5] = 0
        maskCrop[maskCrop > 0.5] = 1

        assert (maskCrop.shape == arrayCrop.shape)
        print("3: ", arrayCrop.mean(), arrayCrop.std())

        masks, patches = Patch_views(arrayCrop, maskCrop, views=1)
        # print(1, patches.shape)

        patchOA = CropZoom(masks, patches * (-1 * masks + 1))
        patchHVV = CropZoom(masks, patches * masks)
        patchHS = CropZoom(masks, patches)
        # print(1, patchOA.shape)

        # patchOA[patchOA < 0] = 0
        # patchHVV[patchHVV < 0] = 0
        # patchHS[patchHS < 0] = 0
        #
        # patchOA[patchOA > 1] = 1
        # patchHVV[patchHVV > 1] = 1
        # patchHS[patchHS > 1] = 1

        patchall = np.concatenate((patchOA, patchHVV, patchHS), axis=-1)

        for view in range(2, 10):
            savePathview = savePath + patientID +".npy"
            masks, patches = Patch_views(arrayCrop, maskCrop, views=view)
            print(view, patches.shape)

            patchOA = CropZoom(masks, patches * (-1 * masks + 1))
            print("4: ", patchOA.mean(), patchOA.std())
            patchHVV = CropZoom(masks, patches * masks)
            patchHS = CropZoom(masks, patches)
            print(view, patchOA.shape)

            # patchOA[patchOA < 0] = 0
            # patchHVV[patchHVV < 0] = 0
            # patchHS[patchHS < 0] = 0
            #
            # patchOA[patchOA > 1] = 1
            # patchHVV[patchHVV > 1] = 1
            # patchHS[patchHS > 1] = 1

            patchall = np.concatenate((patchall,patchOA, patchHVV, patchHS), axis=-1)

            print("4: ", patchall.mean(), patchall.std())

            # np.save(saveOA, patchOA)
            # np.save(saveHVV, patchHVV)
            # np.save(saveHS, patchHS)

            np.save(savePathview, patchall)





