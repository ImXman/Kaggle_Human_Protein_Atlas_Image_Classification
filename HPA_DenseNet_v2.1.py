# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 19:16:19 2020

@author: Yang Xu

DenseNet with combined center loss and competing ratio loss for
Human Protein Atlass classification task
"""

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
#from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense,Input,Conv2D

import warnings
warnings.filterwarnings("ignore")

def load_image(path):
    
    files = "train/"+path
    R = Image.open(files + '_red.png')
    G = Image.open(files + '_green.png')
    B = Image.open(files + '_blue.png')
    Y = Image.open(files + '_yellow.png')
    
    im = np.stack((
        np.array(R), 
        np.array(G), 
        np.array(B),
        np.array(Y)), -1)
        
    im = cv2.resize(im, (512, 512))
    im = np.divide(im, 255)
    im = tf.image.random_crop(im,size=[224,224,4])
    im = tf.constant(im).numpy()
    return im

def CRL(yTrue,yPred):
    '''
    competing ratio loss
    dicrimitive function
    '''
    tp = K.clip(K.sum(yTrue*yPred,axis=1),0.0,1)
    cel = -K.log(tp)
    ##beta = 1 and alpha = 1.5
    np = K.cast(K.clip(K.sum((1-yTrue)*yPred,axis=1),0.0,1), 'float')
    ratio = 1*K.log(1.5+np)
    crl = cel+ratio
    #crl = K.mean(cel+ratio,axis=0)#+K.epsilon()
    return K.mean(crl)

def f1(yTrue,yPred):
    #y_pred = K.round(y_pred)
    yPred = K.cast(K.greater(K.clip(yPred, 0, 1), 0.1), K.floatx())##Threshold 0.1
    tp = K.sum(K.cast(yTrue*yPred, 'float'), axis=0)
    #tn = K.sum(K.cast((1-yTrue)*(1-yPred), 'float'), axis=0)
    fp = K.sum(K.cast((1-yTrue)*yPred, 'float'), axis=0)
    fn = K.sum(K.cast(yTrue*(1-yPred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

class CenterLossLayer(Layer):

    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(28, 128),##Feature dim
                                       initializer='uniform',
                                       trainable=False)
        super().build(input_shape)

    def call(self, x, mask=None):

        # x[0] is Nx128, x[1] is Nx28 onehot, self.centers is 28x128
        distance2center = K.transpose(K.transpose(K.dot(x[1], self.centers))/K.sum(x[1],axis=1)) - x[0]
        delta_centers = K.transpose(K.transpose(K.dot(K.transpose(x[1]),distance2center))/K.sum(K.transpose(x[1]),axis=1))
        center_counts = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1
        delta_centers /= center_counts
        new_centers = self.centers - self.alpha * delta_centers
        self.add_update((self.centers, new_centers), x)
        
        self.result = x[0] - K.transpose(K.transpose(K.dot(x[1], self.centers))/K.sum(x[1],axis=1))
        self.result = K.sum(self.result ** 2, axis=1, keepdims=True)
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

def center_loss(y_true, y_pred):
    return 0.5 * K.sum(y_pred, axis=0)

##-------create model----------------------------------------------------------
densenet = keras.applications.densenet.DenseNet121(include_top=True, weights='imagenet', 
                                                   input_tensor=None, input_shape=None, 
                                                   pooling=None, classes=1000)

densenet = Model(densenet.input, densenet.layers[-2].output)
inputs = Input(shape=(224,224,4), name='Input_layer')
label_inputs = Input(shape=(28,), name='Input_label')
x = Conv2D(3, kernel_size=(1, 1),strides=(1, 1), 
           padding='valid',activation="relu")(inputs)
x = densenet(x)
x = Dense(128,activation="relu")(x)
outputs = Dense(28,activation="softmax",name="Classifier")(x)
centers = CenterLossLayer(alpha=0.5,name="Centerloss")([x, label_inputs])
densenet_hpa = Model(inputs=[inputs,label_inputs], outputs=[outputs,centers])
densenet_hpa.load_weights('./best_checkpoint')
densenet_hpa.compile(optimizer="adam",
                     loss=[CRL,center_loss],
                     loss_weights=[1, 0.0001],
                     metrics=[f1])

##-------preprocess data-------------------------------------------------------
y = pd.read_csv("HPA_train.csv")
y = y.sample(frac=0.2)
ids = y['Id'].values.tolist()
targets = y['Target'].values.tolist()
def split(string):
    return string.split()
targets = list(map(split,targets))
def create_target(values):
    values = [int(i) for i in values]
    vector = np.zeros((28))
    vector[values]=1
    return vector
target_value = list(map(create_target,targets))
target_value = np.asarray(target_value)
ids, X_valid, target_value, y_valid = train_test_split(ids, 
                                                       target_value, 
                                                       test_size=0.2)

class_weights = np.sum(target_value,axis=0)
class_weights = class_weights/np.max(class_weights)
class_weights = np.rint(1/class_weights)

multi_id = pd.DataFrame(ids)[np.sum(target_value,axis=1)>1][0].values.tolist()
multi_target = target_value[np.sum(target_value,axis=1)>1]
uni_id = pd.DataFrame(ids)[np.sum(target_value,axis=1)==1][0].values.tolist()
uni_target = target_value[np.sum(target_value,axis=1)==1]

resamples = np.max(uni_target*class_weights,axis=1).astype("int")
new_uni_id = np.repeat(uni_id, resamples)
new_uni_target = np.repeat(np.argmax(uni_target,axis=1), resamples)
def create_target(values):
    vector = np.zeros((28))
    vector[values]=1
    return vector
new_target_value = list(map(create_target,new_uni_target))
new_target_value = np.asarray(new_target_value)

X_train = multi_id+new_uni_id.tolist()
y_train = np.concatenate((multi_target,new_target_value),axis=0)

del class_weights,ids,multi_id,multi_target,new_target_value,new_uni_id,new_uni_target
del resamples,y,uni_id,uni_target,target_value,targets

##-------Training--------------------------------------------------------------
batch_size = 64

ypred = np.zeros((len(X_valid),28))
iterations = len(X_valid)//batch_size
for k in range(iterations+1):
    image_valid = list(map(load_image,X_valid[k*batch_size:(k+1)*batch_size]))
    image_valid = np.asarray(image_valid).astype("float64")
    pred, _ = densenet_hpa.predict([image_valid,y_valid])
    ypred[k*batch_size:(k+1)*batch_size,:]=pred
F1s = {}
for k in range(28):
    F1s[k]={}
    for i in range(1,1000):
        y = np.greater(ypred[:,k],i/1000).astype("float")
        tp = np.sum(y_valid[:,k]*y)
        fp = np.sum((1-y_valid[:,k])*y)
        fn = np.sum(y_valid[:,k]*(1-y))
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2*p*r / (p+r)
        F1s[k][i]=f1
F1s = pd.DataFrame(F1s)
F1s = F1s.max(axis=0).values
F1s = F1s[np.isfinite(F1s)]
F1s = np.mean(F1s)
old_val_f1 = F1s

y_train = pd.DataFrame(y_train)
y_train['Id']=X_train
epochs=20
image_valid = list(map(load_image,X_valid))
image_valid = np.asarray(image_valid).astype("float32")
valid_f1 = []
for i in range(epochs):
    y_train = y_train.sample(frac=1)
    iterations = y_train.shape[0]//batch_size
    for j in range(iterations):
        suby = y_train.iloc[j*batch_size:(j+1)*batch_size,:28].values.astype("float32")
        subX = y_train.iloc[j*batch_size:(j+1)*batch_size,-1].values.tolist()
        images = list(map(load_image,subX))
        images = np.asarray(images).astype("float64")
        densenet_hpa.fit([images,suby],[suby,np.zeros((images.shape[0], 1))],verbose=0)
    #scores = densenet_hpa.evaluate([image_valid,y_valid],
    #                               [y_valid,np.zeros((image_valid.shape[0], 1))],
    #                               verbose=0)
    ypred = np.zeros((len(X_valid),28))
    iterations = len(X_valid)//batch_size
    for k in range(iterations+1):
        image_valid = list(map(load_image,X_valid[k*batch_size:(k+1)*batch_size]))
        image_valid = np.asarray(image_valid).astype("float64")
        pred, _ = densenet_hpa.predict([image_valid,y_valid])
        ypred[k*batch_size:(k+1)*batch_size,:]=pred
    F1s = {}
    for k in range(28):
        F1s[k]={}
        for i in range(1,1000):
            y = np.greater(ypred[:,k],i/1000).astype("float")
            tp = np.sum(y_valid[:,k]*y)
            fp = np.sum((1-y_valid[:,k])*y)
            fn = np.sum(y_valid[:,k]*(1-y))
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f1 = 2*p*r / (p+r)
            F1s[k][i]=f1
    F1s = pd.DataFrame(F1s)
    F1s = F1s.max(axis=0).values
    F1s = F1s[np.isfinite(F1s)]
    F1s = np.mean(F1s)
    #print("Validation F1: "+str(round(scores[3],5)))
    print("Validation F1: "+str(round(F1s,5)))
    #valid_f1.append(round(scores[3],5))
    valid_f1.append(round(F1s,5))
    #if np.isfinite(scores[3]) and scores[3]>=old_val_f1:
    if np.isfinite(F1s) and F1s>=old_val_f1:
        densenet_hpa.save_weights('./best_checkpoint')
        old_val_f1 = F1s#scores[3]
    K.clear_session()
        
np.savetxt("Validation_F1.txt",np.array(valid_f1),delimiter="\t")

#densenet_hpa.load_weights('./best_checkpoint')
#ypred, _ = densenet_hpa.predict([image_valid,y_valid])

##fine tuning
#F1s = {}
#for k in range(28):
#    F1s[k]={}
#    for i in range(1,1000):
#        y = np.greater(ypred[:,k],i/1000).astype("float")
#        tp = np.sum(y_valid[:,k]*y)
#        fp = np.sum((1-y_valid[:,k])*y)
#        fn = np.sum(y_valid[:,k]*(1-y))
#        p = tp / (tp + fp)
#        r = tp / (tp + fn)
#        f1 = 2*p*r / (p+r)
#        F1s[k][i]=f1
#F1s = pd.DataFrame(F1s)
#F1s.to_csv("densenet_Validation_F1_finetune.csv")
#F1s = F1s.max(axis=0).values
#F1s = F1s[np.isfinite(F1s)]
#print(np.mean(F1s))