# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 01:08:05 2019

@author: war19
"""


import numpy as np

from keras import layers
from keras.models import Model
from keras.models import load_model
import cv2
import os
import string

import matplotlib.pyplot as plt

captcha_symbol = string.ascii_lowercase+"0123456789"
num_captcha_symbol = len(captcha_symbol)
img_shape=(50,200,1)

def create_model():
    img=layers.Input(shape=img_shape)
    conv1=layers.Conv2D(16,(3,3), padding='same',activation='relu')(img)
    mp1=layers.MaxPooling2D(padding='same')(conv1)
    conv2=layers.Conv2D(32,(3,3), padding='same',activation='relu')(mp1)
    mp2=layers.MaxPooling2D(padding='same')(conv2)
    conv3=layers.Conv2D(16,(3,3), padding='same',activation='relu')(mp2)
    bn = layers.BatchNormalization()(conv3)
    mp3=layers.MaxPooling2D(padding='same')(bn)
    flat=layers.Flatten()(mp3)
    outs=[]
    for _ in range(5):
        dens1=layers.Dense(64,activation='relu')(flat)
        drop=layers.Dropout(0.5)(dens1)
        res = layers.Dense(num_captcha_symbol,activation='sigmoid')(drop)
        outs.append(res)
        
    model=Model(img,outs)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=["accuracy"])
    return model
def preprocess_data():
    n_samples=len(os.listdir('samples'))
    x=np.zeros((n_samples, 50, 200, 1))
    y=np.zeros((5,n_samples,num_captcha_symbol))
    for i,pic in enumerate(os.listdir('samples')):
        img=cv2.imread(os.path.join('samples',pic),cv2.IMREAD_GRAYSCALE)
        pic_target = pic[:-4]
        if len(pic_target)<6:
            img=img/255.0
            img=np.reshape(img,(50,200,1))
            targs=np.zeros((5,num_captcha_symbol))
            for j,l in enumerate(pic_target):
                ind  =captcha_symbol.find(l)
                targs[j,ind]=1
            x[i]=img
            y[:,i]=targs
    return x,y 
x,y = preprocess_data()
x_train,y_train=x[:970],y[:,:970]
x_test,y_test = x[970,:],y[:,:970:]

model=create_model();
model.summary();

hist = model.fit(x_train, [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]], batch_size=32, epochs=30,verbose=1, validation_split=0.2)



import matplotlib.pyplot as plt
img=cv2.imread('samples/k.png',cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap=plt.get_cmap('gray'))

print('predicted captcha is :',predict('samples/k.png'))


            