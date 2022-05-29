
import opendatasets as od

###Downloading dataset

od.download("https://www.kaggle.com/deadskull7/fer2013")

import os, shutil

if os.path.exists('data'):
    pass
else:
    os.mkdir('data')

shutil.move('fer2013/fer2013.csv', 'data')
os.rmdir('fer2013')

"""Importing all Necessary Libraries"""

import sys,os

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten  
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D,GlobalAveragePooling2D
from tensorflow.keras.losses import categorical_crossentropy  
from tensorflow.keras.optimizers import Adam  
from tensorflow.keras.regularizers import l2 
from tensorflow.keras import utils

#checking whether the gpu is working fine

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)

"""###Data pre-processing"""

#reading file and extracting data

emotion_data = pd.read_csv('data/fer2013.csv')
INTERESTED_LABELS=[0,3,4,6]
emotion_data=emotion_data[emotion_data.emotion.isin(INTERESTED_LABELS)]
le = LabelEncoder()#label encoder as we take 4 out of the 7 emotion
emotion_data.emotion = le.fit_transform(emotion_data.emotion)

# splitting the data into train and test sets

X_train,train_y,X_test,test_y=[],[],[],[]  
for index, row in emotion_data.iterrows():  
    val=row['pixels'].split(" ")  
    if 'Training' in row['Usage']:
      X_train.append(np.array(val,'float32'))  
      train_y.append(row['emotion'])  
    elif 'PublicTest' in row['Usage']:  
      X_test.append(np.array(val,'float32'))  
      test_y.append(row['emotion'])
        
num_features = 64  
num_labels = 4

emotion_labels = ["Angry", "Happy", "Sad", "Neutral"]
num_classes = len(emotion_labels)

batch_size = 64  
epochs = 100
width, height = 48, 48 

#Converting the image into an array of pixel numbers

X_train = np.array(X_train,'float32')                                                         
train_y = np.array(train_y,'float32')                                                                   
X_test = np.array(X_test,'float32')                                                       
test_y = np.array(test_y,'float32')                                                   
train_y= utils.to_categorical(train_y, num_classes=4)   # matching the pixel array to specified emotion label in train set              
test_y= utils.to_categorical(test_y, num_classes=4)     # matching the pixel array to specified emotion label in test set                    
X_train -= np.mean(X_train, axis=0)                                                 
X_train /= np.std(X_train, axis=0)                                                    
X_test -= np.mean(X_test, axis=0)                                                                   
X_test /= np.std(X_test, axis=0)                                                      
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)                                              
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)                                             


"""

###Building Model
"""

num_samples, num_classes = emotion_data.shape
#Creating a CNN
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same', name='image_array', input_shape=(X_train.shape[1:])))
model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same'))
model.add(BatchNormalization())  #using batch normalization to help in gradient descent
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(.3))               # adding dropout for regularization

model.add(Conv2D(filters=96, kernel_size=(5, 5), padding='same')) #using same padding to retain the size of the image for easier processing
model.add(Conv2D(filters=96, kernel_size=(5, 5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(.3))

model.add(Conv2D(filters=128, kernel_size=(5, 5), padding='same'))
model.add(Conv2D(filters=128, kernel_size=(5, 5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(.3))
model.add(Conv2D(filters=256, kernel_size=(5, 5), padding='same'))
model.add(Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu'))  
model.add(BatchNormalization())  
model.add(GlobalAveragePooling2D())  

model.add(Flatten()) 

# Fully connected layer 1st layer

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(num_labels, activation='sigmoid'))

model.compile(loss=categorical_crossentropy,  
              optimizer=Adam(),  
              metrics=['accuracy'])  

model.summary()

#using early stopping to reduce overfitting
cb_early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    min_delta=0.0005,
    patience=11,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)
#reducing learning rate when the gradient descent slows down
cb_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_accuracy",
    factor=0.5,
    patience=10,
    verbose=1,
    mode="auto",
    min_lr=1e-7,
)

#Real time data augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1.0/255.0, rotation_range=20,
                                 height_shift_range=0.1,
                                 width_shift_range=0.4,
                                 shear_range=0.3,horizontal_flip=True,
                                 zoom_range=0.2
                                )
validation_datagen=ImageDataGenerator(rescale=1.0/255.0) #scaling the input image
train_datagen.fit(X_train)

model.fit(train_datagen.flow(X_train, train_y, batch_size=batch_size),
       
         validation_data=validation_datagen.flow(X_test, test_y,
         batch_size=batch_size),
         steps_per_epoch=len(X_train) / batch_size, epochs=epochs, callbacks=[cb_early_stop,cb_reduce_lr])

model.compile(loss='sparse_categorical_crossentropy',
         optimizer=Adam(),
         metrics=['accuracy'])

fer_json = model.to_json()  
with open("fer.json", "w") as json_file:  
    json_file.write(fer_json)  
model.save_weights("fer.h5")

#Testing the model on new images
score = model.predict(X_test)

pred_y = [np.argmax(item) for item in score]
true_y = [np.argmax(item) for item in test_y]

accuracy = [(x==y) for x,y in zip(pred_y,true_y)]
print("accuracy on testset: " , np.mean(accuracy))

