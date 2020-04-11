# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 20:35:14 2020

@author: xuyan

Keras DenseNet
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
#from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense,Input,Conv2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

def load_image(path,label):
    files = "human-protein-atlas-image-classification/train/"+path
    f = files + '_red.png'
    R = tf.io.read_file(f)
    R = tf.image.decode_png(R, channels=1)
    R = tf.image.convert_image_dtype(R, tf.float32)
    R = tf.image.resize(R, [512, 512])
    R = tf.reshape(R, [512,512]) 
    
    f = files + '_green.png'
    G = tf.io.read_file(f)
    G = tf.image.decode_png(G, channels=1)
    G = tf.image.convert_image_dtype(G, tf.float32)
    G = tf.image.resize(G, [512, 512])
    G = tf.reshape(G, [512,512]) 
    
    f = files + '_blue.png'
    B = tf.io.read_file(f)
    B = tf.image.decode_png(B, channels=1)
    B = tf.image.convert_image_dtype(B, tf.float32)
    B = tf.image.resize(B, [512, 512])
    B = tf.reshape(B, [512,512]) 
    
    f = files + '_yellow.png'
    Y = tf.io.read_file(f)
    Y = tf.image.decode_png(Y, channels=1)
    Y = tf.image.convert_image_dtype(Y, tf.float32)
    Y = tf.image.resize(Y, [512, 512])
    Y = tf.reshape(Y, [512,512]) 
    
    im = tf.stack([R,G,B,Y], -1)
    
    im = im/255
    return im,label

def augment(image,label):
    image = tf.image.random_crop(image,size=[224,224,4])
    return image,label

def CRL(yTrue,yPred):
    '''
    competing ratio loss
    dicrimitive function
    '''
    cel = -K.sum(yTrue*K.log(yPred),axis=1)
    ##beta = 1 and alpha = 1.5
    ratio = 1*K.log(1.5+K.sum(K.abs(yTrue-1)*yPred,axis=1))
    #crl = cel+ratio
    crl = K.mean(cel+ratio,axis=0)
    return crl

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

densenet = keras.applications.densenet.DenseNet121(include_top=True, weights='imagenet', 
                                                   input_tensor=None, input_shape=None, 
                                                   pooling=None, classes=1000)

densenet = Model(densenet.input, densenet.layers[-2].output)
inputs = Input(shape=(224,224,4), name='Input_layer')
x = Conv2D(3, kernel_size=(1, 1),strides=(1, 1), 
           padding='valid',activation="relu")(inputs)
x = densenet(x)
outputs = Dense(28,activation="softmax")(x)
densenet_hpa = Model(inputs, outputs)

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-2
    if epoch > 300:
        lr *= 0.5e-3
    elif epoch > 200:
        lr *= 1e-3
    elif epoch > 100:
        lr *= 1e-2
    elif epoch > 50:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

checkpoint = ModelCheckpoint(filepath="./best_model.hdf5",
                             monitor='val_loss',
                             verbose=1,mode='min',
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

densenet_hpa.compile(optimizer='adam',
                     loss=CRL,
                     metrics=[f1])

y = pd.read_csv("HPA_train.csv")
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

X_train, X_test, y_train, y_test = train_test_split(ids, target_value, test_size=0.15)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15)

Train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
Valid = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))

batch_size = 64
num_train_samples = len(X_train)
AUTOTUNE = tf.data.experimental.AUTOTUNE

augmented_train_batches = (
    Train
    .take(num_train_samples)
    .cache()
    .shuffle(num_train_samples//4)
    .map(load_image,num_parallel_calls=AUTOTUNE)
    .map(augment,num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
    )

validation_batches = (
    Valid
    .map(load_image,num_parallel_calls=AUTOTUNE)
    .map(augment,num_parallel_calls=AUTOTUNE)
    .batch(2*batch_size)
    )

history = densenet_hpa.fit(augmented_train_batches, epochs=100,
                           validation_data=validation_batches,
                           callbacks = [checkpoint,lr_reducer, lr_scheduler])

#history = pd.DataFrame(history)
#history.to_csv("training_history.csv")
np.savetxt("test_filenames.txt",np.array(X_test),delimiter="\t",fmt="%s")
np.savetxt("test_target.txt",np.array(y_test),delimiter="\t",fmt="%i")
#X_test = None
#model = load_model("best_model.hdf5",custom_objects={'CRL': CRL})
#y_pred = model.predict(X_test)
