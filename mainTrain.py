import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical

image_directory='datasets/'
# print a list of names of all the files present in the specified path.
no_tumor_images=os.listdir(image_directory+ 'no/') 
yes_tumor_images=os.listdir(image_directory+ 'yes/')

dataset=[]
label=[]
INPUT_SIZE=64 #Specify the size that we want to resize height and width of the image to

## Exploratory Data Analysis- Class Balance, Size

##Preprocessing
for i, image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1]=='jpg'): #Check that it is a jpg file
        image=cv2.imread(image_directory+'no/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)



for i, image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1]=='jpg'): #Check that it is a jpg file
        image=cv2.imread(image_directory+'yes/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)


#Check that lengths are 3000 corresponding to 3000 images labelled yes and no
print(len(dataset))
print(len(label))

#Convert to np array as train_test_split requires that
dataset=np.array(dataset)
label=np.array(label)

#From EDA, we have no class imbalance

x_train, x_test, y_train, y_test= train_test_split(dataset, label, test_size=0.2, random_state=0) 
#random state ensures that split is replicable

# print(x_train.shape) # 2400 training eg, 64 , 64, 3 channels-RGB
# print(y_train.shape)

# print(x_test.shape) # 600 training eg, 64 , 64, 3 channels-RGB
# print(y_test.shape)


#Normalize with normalize fn from Keras
x_train=normalize(x_train, axis=1) 
x_test=normalize(x_test, axis=1)
print(x_test[0,:,:,0])
y_train=to_categorical(y_train, num_classes=2)
y_test=to_categorical(y_test, num_classes=2)

#Check Normalisation 
# def visualize_tensor(tensor, batch_size, channels, sample_index):
#     if len(tensor.shape) != 4:
#         print("Input tensor must have 4 dimensions: (batch_size, height, width, channels)")
#         return
    
#     if tensor.shape[3] != channels:
#         print(f"Input tensor must have {channels} channels")
#         return
    
#     if batch_size < 1 or batch_size > tensor.shape[0]:
#         print(f"Batch size should be between 1 and {tensor.shape[0]}")
#         return
    
#     if sample_index < 0 or sample_index >= batch_size:
#         print(f"Sample index should be between 0 and {batch_size - 1}")
#         return

#     plt.figure(figsize=(5, 5))
#     plt.imshow(tensor[sample_index])
#     plt.title(f"Image {sample_index + 1}")
#     plt.axis('off')
#     plt.show()


# sample_index = 0  # Choose the specific data sample you want to visualize

# visualize_tensor(x_test, batch_size=600, channels=3, sample_index=2)


##Custom Model Building
model= Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(INPUT_SIZE,INPUT_SIZE,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=32, kernel_size=(3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=32, kernel_size=(3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2)) #our output is only 2-binary classifier
model.add(Activation('sigmoid'))

#Compile by specifying loss, optimizer and metric
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=16,verbose=1, epochs=10, 
          validation_data=(x_test,y_test),shuffle=False)

#save trained model
model.save('BrainTumor10epochsCategorical.h5')