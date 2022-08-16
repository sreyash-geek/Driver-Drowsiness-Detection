import os
from keras.preprocessing import image
import matplotlib.pyplot as plt 
import numpy as np
from keras.utils.np_utils import to_categorical
import random,shutil
from keras.models import Sequential
from keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D, BatchNormalization
from keras.models import load_model


def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True,batch_size=1,target_size=(24,24),class_mode='categorical' ):

    return gen.flow_from_directory(dir,batch_size=batch_size,shuffle=shuffle,color_mode='grayscale',class_mode=class_mode,target_size=target_size)


BATCH_SIZE = 32
TARGET_SIZE = (24,24)


train_batch= generator("C:\\Users\\SREYASH\\Desktop\\Drowsiness detection\\archive\\dataset_new\\train",shuffle=True, batch_size=BATCH_SIZE,target_size=TARGET_SIZE)
valid_batch= generator("C:\\Users\\SREYASH\\Desktop\\Drowsiness detection\\archive\\dataset_new\\test",shuffle=True, batch_size=BATCH_SIZE,target_size=TARGET_SIZE)
STEPS_PER_EPOCH= len(train_batch.classes)//BATCH_SIZE
VALIDATION_SIZE = len(valid_batch.classes)//BATCH_SIZE
print(STEPS_PER_EPOCH,VALIDATION_SIZE)


model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24,24,1)),
    MaxPooling2D(pool_size=(1,1)),
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(1,1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1,1)),  
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


model.fit(train_batch, validation_data=valid_batch,epochs=15,steps_per_epoch=SPE ,validation_steps=VS)


model.save('models/DrowsinessDetection.h5', overwrite=True)