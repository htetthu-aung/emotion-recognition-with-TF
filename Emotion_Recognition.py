#!/usr/bin/env python
# coding: utf-8

# In[60]:


import sys
import os
import numpy as np
from keras.models import sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D,BatchNormalization
from keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils
import pandas as pd


# In[61]:


df = pd.read_csv("D:\\Projects\\Emotion Recognition\\emotion recognition dataset\\fer2013.csv")


# In[62]:


df.info()


# In[63]:


df.describe()


# In[64]:


df.head(5)


# In[65]:


print(df["Usage"].value_counts())


# In[66]:


X_train, X_test, y_train, y_test = [],[],[],[]

for index,row in df.iterrows():
    val = row["pixels"].split(" ")
    try:
        if "Training" in row["Usage"]:
            X_train.append(np.array(val,"float32"))
            y_train.append(row["emotion"])
            
        elif "PublicTest" in row["Usage"]:
            X_test.append(np.array(val,"float32"))
            y_test.append(row["emotion"])
            
    except:
        print(f"error occured at index: {index} and row: {row}")
        
        
print(f"X_train sample data:{X_train[0:2]}")
print(f"X_test sample data: {X_test[0:2]}")
print(f"y_train sample data: {y_train[0:2]}")
print(f"y_test sample data: {y_test[0:2]}")


# In[67]:


X_train = np.array(X_train,"float32")
y_train = np.array(y_train,"float32")
X_test = np.array(X_test,"float32")
y_test = np.array(y_test,"float32")


# In[68]:


#Normalizing data between 0 and 1

X_train -= np.mean(X_train, axis=0)
X_train /= np.std(X_train, axis=0)

X_test -= np.mean(X_test, axis=0)
X_test /= np.std(X_test, axis=0)


# In[69]:


num_features = 64
num_labels = 7
batch_size = 64
epochs = 30
width,height = 48,48

X_train = X_train.reshape(X_train.shape[0],width, height,1)

X_test = X_test.reshape(X_test.shape[0],width, height,1)


# In[71]:


X_train.shape


# In[72]:


X_train[10].shape#after reshape


# In[73]:


#Designing in CNN

model = sequential.Sequential()

#1st layer
model.add(Conv2D(num_features, kernel_size=(3,3),activation="relu",input_shape =(X_train.shape[1:])))
model.add(Conv2D(num_features, kernel_size=(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.5))


#2nd convolutional layer
model.add(Conv2D(num_features, kernel_size=(3,3),activation="relu"))
model.add(Conv2D(num_features, kernel_size=(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.5))

#3rd convolutional layer
model.add(Conv2D(2*num_features, kernel_size=(3,3),activation="relu"))
model.add(Conv2D(2*num_features, kernel_size=(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())

model.add(Dense(2*2*2*2*num_features,activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(num_labels,activation="softmax"))

model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(),metrics=['accuracy'])



# In[74]:


model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_test,y_test),shuffle=True)


# fer_json = model.to_json()
# with open("fer.json","w") as json_file:
#     json_file.write(fer_json)

# model.save_weights("fer.h5")


fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("fer.h5")