#%%
import numpy as np
import cv2
#%%
def loadLabelData(path,skiprows,mode = 'train'):
    if mode == 'train':
        X,y = np.loadtxt(path, dtype = str, delimiter = ',',skiprows=skiprows,unpack = True)
        return X,y
    else:
        X = np.loadtxt(path, dtype = str, delimiter = ',',skiprows=skiprows,unpack = True)
        return X
#%%
class_labels = ['antelope','bat','beaver','bobcat','buffalo','chihuahua','chimpanzee','collie','dalmatian','german+shepherd','grizzly+bear','hippopotamus','horse','killer+whale','mole','moose','mouse','otter','ox','persian+cat','raccoon','rat','rhinoceros','seal','siamese+cat','spider+monkey','squirrel','walrus','weasel','wolf']
def classLabels():
    return class_labels
#%%
def convertOutputArray(y):
   output = np.zeros((len(class_labels),len(y)))
   for i in range(len(class_labels)):
       output[i][:] = (y == class_labels[i])
   return output.T
#%%
def getInputFeatures(m,folderPath,display=True):
    newX = np.zeros((m,128,128),dtype = 'float')
    percent = 0
    ignored = list()
    for i in range(1,m+1):
        imgName = 'Img-'+str(i)+'.jpg'
        img = cv2.imread(folderPath+'/'+imgName,0)
        if type(img) == type(None):
            ignored.append(i)
            continue
        img = cv2.resize(img,(128,128))
        newX[i-1] = img
        if display:
            percent = i*100/m
            print('\r{0:.2f}'.format(percent),end=' ',flush = True)
    return newX,ignored
#%%
def saveData(X_feature,y_feature = None,mode = 'train'):
    if mode == 'train':
        np.save('X_train',X_feature)
        np.save('y_train',y_feature)
    if mode == 'test':
        np.save('X_test',X_feature)
#%%
def loadData(mode = 'train'):
    if mode == 'train':
        X_features = np.load('X_train.npy')
        y_features = np.load('y_train.npy')
        return X_features,y_features
    if mode == 'test':
        X_features = np.load('X_test.npy')
        return X_features
'''
#%%
print(X_train.shape)
print(y_train.shape)
'''