#%%
#IMPORTING MODULES

import tensorflow as tf
import numpy as np
from dataPreprocess import loadLabelData,getInputFeatures,convertOutputArray,loadData,saveData

#%%
def downloadDataset(labelPath = None,imgFolderPath = None,mode = 'train',init = False):
    if mode == 'train':
        if init == True:
            if labelPath == None:
                print('Please provide metadata path')
                labelPath = input()
            X_labels,y_labels = loadLabelData(labelPath,skiprows = 1, mode = mode)
            m = len(X_labels)
            print(m)
            if imgFolderPath == None:
                print('Please provide imgFolder path')
                imgFolderPath = input()
            print('Initializing Datasets')
            X_features,ignored = getInputFeatures(m,imgFolderPath)
            print(ignored)
            y_features = convertOutputArray(y_labels)
            saveData(X_features,y_features,mode = mode)
        elif init == False:
            print('Loading Previous Datasets')
            X_features,y_features = loadData(mode)
        print('Done')
        return X_features,y_features

    elif mode == 'test':
        if init == True:
            if labelPath == None:
               print('Please provide metadata path')
               labelPath = input()
            X_tlabels = loadLabelData(labelPath,skiprows = 1,mode = mode)
            m = len(X_tlabels)
            print(m)
            if imgFolderPath == None:
                print('Please provide imgFolder path')
                imgFolderPath = input()
            print('Initializing Datasets')
            X_features,ignored = getInputFeatures(m,imgFolderPath)
            print(ignored)
            saveData(X_features,mode = mode)
        elif init == False:
            print('Loading Previous Datasets')
            X_features = loadData(mode)
        print('Done')
        return X_features
#READING DATA

#%%
print('First time Run?(Y/N)')
c = input()
if c == 'Y' or c == 'y':
    init = True
else:
    init = False 
X_train,y_train = downloadDataset(labelPath='./DL_Beginner/meta-data/train.csv',imgFolderPath='./DL_Beginner/train',mode = 'train',init=init)
X_test = downloadDataset(labelPath='./DL_Beginner/meta-data/test.csv',imgFolderPath='./DL_Beginner/test',mode = 'test',init= init)
#data preproscessing
print(X_train.shape,y_train.shape)
print(X_test.shape)