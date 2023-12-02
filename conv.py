# -*- coding: utf-8 -*-

#code to import drive 
from google.colab import drive
drive.mount('/content/drive')

#import tensorflow and os
import tensorflow as tf
import os

# config GPU
gpus=tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu,True)

#import cv2 and imghdr
import cv2
import imghdr

#loding the data path to the data_dir var
#load your location of the data set

data_dir='/content/drive/MyDrive/'

#this are the type of extension support in python
image_list=['jpeg','jpg','bmp','png']

#just view the images in the data_dir
for i in os.listdir(data_dir):
  for j in os.listdir(os.path.join(data_dir,i)):
    print(j)

#make clean dataset by removing the unsuported formate in the dataset
for image_clf in os.listdir(data_dir):
  for image in os.listdir(os.path.join(data_dir, image_clf)):
    image_path=os.path.join(data_dir,image_clf,image)
    try:
      img=cv2.imread(image_path)
      tip=imghdr.what(image_path)
      if tip not in image_list:
        print("Image not in ext list {}".format(image_path))
        os.remove(image_path)
    except Exception as e:
      print("Issue {}",format(image_path))

#helpcode suport in colab
# tf.data.Dataset??

#import numpy and plt 
import numpy as np
from matplotlib import pyplot as plt

#load the data
data=tf.keras.utils.image_dataset_from_directory('/content/drive/MyDrive/')

data_itt=data.as_numpy_iterator()

#creat the batch
batch=data_itt.next()

batch[0].shape

# """# **Perprocessing**"""

scaled_data=data.map(lambda x, y:(x/255, y))

scaled_itt = scaled_data.as_numpy_iterator()

batch=scaled_itt.next()

batch[0].max()

# """VIZ the data after pre process"""

fig, ax =plt.subplots(ncols=4,figsize=(20,20))
for idx,img in enumerate(batch[0][:4]):
  ax[idx].imshow(img)
  ax[idx].title.set_text(batch[1][idx])

# """Spliting the data set"""

len(data)

train_size=int(len(data)*.8)-1
val_size=int(len(data)*.1)+1
test_size=int(len(data)*.1)+1

train_size,val_size,test_size

train_size+val_size+test_size

# """take data to train, val, test"""

train=data.take(train_size)
val=data.skip(train_size).take(val_size)
test=data.skip(train_size+val_size).take(test_size)

# """Train model"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

model= Sequential()

model.add(Conv2D(16,(3,3),1,activation='relu',input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3),1,activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16,(3,3),1,activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile( 'adam' ,loss=tf.losses.BinaryCrossentropy(),metrics=['accuracy'])

model.summary()

# """callback"""

logs='/content/drive/MyDrive/logs'

tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir=logs)

hist=model.fit(train,epochs=20,validation_data=val,callbacks=[tensorboard_callback])

fig=plt.figure()
plt.plot(hist.history['loss'],color='green',label='loss')
plt.plot(hist.history['val_loss'],color='red',label='val_loss')
plt.suptitle('loss',fontsize=20)
plt.legend(loc='upper left')
plt.show()

# """Evalve"""

from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

pre=Precision()
rec=Recall()
acc=BinaryAccuracy()

for batch in test.as_numpy_iterator():
  x,y=batch
  y_pred=model.predict(x)
  pre.update_state(y,y_pred)
  rec.update_state(y,y_pred)
  acc.update_state(y,y_pred)

print(f'Precision: {pre.result().numpy():.2f}')
print(f'Recall: {rec.result().numpy():.2f}')
print(f'Accuracy: {acc.result().numpy():.2f}')

import cv2

#read the test image
img=cv2.imread('nc-type.jpg')
#viz the image
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()

#convert the size in to 256,256
resixed=cv2.resize(img,(256,256))
plt.imshow(cv2.cvtColor(resixed,cv2.COLOR_BGR2RGB))
plt.show()

# represent the model to predict the new images
y_pre = model.predict(tf.expand_dims(resixed/255,axis=0))

# y_pre contain the output of the model

# result cal in 0 and 1
if y_pre>0.5:
  print('0')
else:
  print('1')