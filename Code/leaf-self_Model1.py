# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 10:46:11 2019

@author: andy3
"""
#########################################
#
#Leaf-classification
#
#
#
#
#
#
#########################################
import numpy as np
import pandas as pd
from keras import layers 
from keras import models
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils

#Read the data of kaggle's data
train=pd.read_csv('C:/Users/andy3/leaf-classification/train.csv')

test=pd.read_csv('C:/Users/andy3/leaf-classification/test.csv')

print(train)

print(test)
#encode the species to number as label frpm train.csv
label=LabelEncoder().fit_transform(train.species)

print("The label of train.csv:",label)

#extract all type of species from train.csv
classes=LabelEncoder().fit(train.species).classes_

#Put it list[]
classes=list(classes)

print("The list of all species:",classes)

#correspindence between label and classes
classes_label=LabelEncoder().fit_transform(classes)

print("Correspondence between label and classes:",classes_label)

#Drop out of row of "id" and "species" from the train.csv and test.csv
train=train.drop(['id','species'],axis=1)

print(train)

test=test.drop(['id'],axis=1)

print(test)

#Take the the normalization method to re-describe the feature quantity.
scaler=StandardScaler().fit_transform(train.values)

print(scaler.shape)


#Spilt the train_data into X_train, X_val and their labels.
spilt_tra_val = StratifiedShuffleSplit(test_size=0.2, random_state=23)

for train_index, val_index in spilt_tra_val.split(scaler,label):
    
    X_train, X_val = scaler[train_index], scaler[val_index]
    
    X_train_label, X_val_label = label[train_index], label[val_index]

#put the X_train ,X_val into array as input_data of model. 
    
print(X_train.shape)

print(X_val.shape)
nb_features=64

nb_classes=99

input_train=np.zeros((len(X_train),64,3))

input_train[:,:,0]=X_train[:,:64]
input_train[:,:,1]=X_train[:,64:128]
input_train[:,:,2]=X_train[:,128:]

print(input_train)
print(input_train.shape)


input_val=np.zeros((len(X_val),64,3))

input_val[:,:,0]=X_val[:,:64]
input_val[:,:,1]=X_val[:,64:128]
input_val[:,:,2]=X_val[:,128:]
print(input_val)
print(input_val.shape)
model = Sequential()

model.add(layers.Conv1D(256,
                        1,
                        activation='relu',
                        input_shape=(64, 3)))

model.add(Flatten())

model.add(layers.Dense(256,
                       activation='relu'))

 
model.add(layers.Dense(128,
                       activation='relu'
                       )
    )

#The multi-classifical problem,so we use softmax with 46 NN union to achieve the purpose.
x=model.add(layers.Dense(99,
                       activation='softmax'
                       )
    )


#Compile the model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

X_train_label = np_utils.to_categorical(X_train_label, 99)

X_val_label = np_utils.to_categorical(X_val_label, 99)

from keras.utils import plot_model

plot_model(model,show_shapes=True,to_file='leaf_classification.png')    
    

rmsprop = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)

model.compile(loss='categorical_crossentropy',optimizer=rmsprop,metrics=['acc'])

history=model.fit(input_train,
                  X_train_label,
                  epochs=15,
                  validation_data=(input_val, X_val_label),
                  batch_size=16)

from keras.utils import plot_model

plot_model(model,show_shapes=True,to_file='leaf_classificationModel1.png')    
    

import matplotlib.pyplot as plt

acc=history.history['acc']

val_acc=history.history['val_acc']

loss=history.history['loss']

val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)


plt.plot(epochs,
         acc,
         'bo',
         label='4:1 Traning accuracy')

plt.plot(epochs,
         val_acc,
         'b',
         label='4:1 Validation accuracy')

plt.title('The accuracy of training and validation in Model1')

plt.legend()

plt.figure()

plt.plot(epochs,
         loss,
         'bo',
         label="4:1 Training loss quantity")

plt.plot(epochs,
         val_loss,
         'b',
         label='4:1 Validation loss quantity')

plt.title('Training and validation loss in Model1')

plt.legend()








