#!/usr/bin/env python
# coding: utf-8
import argparse
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.utils import class_weight
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import skimage
import itertools

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--output', default='model_custom.hdf5', type=str, 
                    help='Duong dan luu mo hinh')
parser.add_argument('--train_data', default='preprocess_256x256/train2/', type=str, 
                    help='Duong dan den du lieu training')
parser.add_argument('--test_data', default='preprocess_256x256/test2/', type=str, 
                    help='Duong dan den du lieu testing')   
parser.add_argument('--batch_size', default=4, type=int, 
                    help='Batch size') 
parser.add_argument('--epoch', default=15, type=int, 
                    help='Epochs') 

args = parser.parse_args() #so luong anh tinh toan trong 1 batch
num_classes = 4 #4 folder voi du lieu
image_size = 128
batch_size = args.batch_size
epochs = args.epoch
data_path = args.train_data
test_path = args.test_data

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

train_datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255) # set validation split

train_generator = train_datagen.flow_from_directory(data_path, target_size=(image_size, image_size),
                                                    batch_size=batch_size, 
                                                    color_mode='grayscale',
                                                    class_mode='categorical',
                                                    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(data_path, target_size=(image_size,image_size),
                                                         batch_size=batch_size,
                                                         color_mode='grayscale',
                                                         class_mode='categorical',
                                                         subset='validation') # set as validation data


test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_path, target_size=(image_size, image_size),
                                                  batch_size=batch_size,
                                                  color_mode='grayscale',
                                                  class_mode='categorical')


# In[4]:


class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(train_generator.classes),
                                                  train_generator.classes)


model = Sequential()
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', input_shape=(image_size, image_size,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(num_classes, activation='softmax'))

print(model.summary())

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

filepath=args.output
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Train the network
history = model.fit_generator(train_generator,
                              steps_per_epoch = train_generator.samples // batch_size,
                              validation_data = validation_generator,
                              validation_steps = validation_generator.samples // batch_size,
                              epochs = epochs, 
                              callbacks=callbacks_list,
                              class_weight=class_weights
                             )

score = model.evaluate_generator(test_generator,steps = test_generator.samples // batch_size) 
print("\n\n")
print('Test Loss:', score[0])
print('Test accuracy:', score[1])
