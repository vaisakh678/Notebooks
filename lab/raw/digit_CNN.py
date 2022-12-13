#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, Dropout


# In[3]:


mnist = keras.datasets.mnist
#from keras.datasets import mnist


# In[4]:


(x_train_full, y_train_full),(x_test, y_test) = mnist.load_data()
print(x_train_full.shape)


# In[5]:


print("x_train_full", x_train_full.shape)
print("y_train_full", y_train_full.shape)
print("x_test", x_test.shape)
print("y_test", y_test.shape)


# In[6]:


#show first train and test image
plt.imshow(x_train_full[0])
print(y_train_full[0])


# In[7]:


plt.imshow(x_test[0])
print(y_test[0])


# In[8]:


#find the number of train data and test data?
len(x_train_full)


# In[9]:


len(y_train_full)


# In[10]:


len(x_test)


# In[11]:


len(y_test)


# In[12]:


#print the first train data
print(x_train_full[0])


# In[13]:


#find dimension of train and test data
print(x_train_full.shape)
print(y_train_full.shape)
print(x_test.shape)
print(y_test.shape)


# In[14]:


#normalising
x_train_norm = x_train_full/255.
x_test_norm = x_test/255.
print(x_train_norm.shape)
print(x_test_norm.shape)


# In[15]:


print(x_train_norm[0])


# In[16]:


X_train = x_train_norm.reshape(-1,28,28,1)   
X_test = x_test_norm.reshape(-1,28,28,1)
print(X_train.shape)
print(X_test.shape)
#-1 no of datas
#28 pixels row and cols
# 1 channels
#cnn in 4 dimension


# In[17]:


y_train_full[0]


# In[18]:


# Find the unique numbers from the train labels
classes = np.unique(y_train_full)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)


# In[19]:


x_valid, x_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
#


# In[20]:


# Building model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1))) #kernal(filter_intializer for activation)
model.add(Dropout(0.25)) #25% of neurons are removed from the model
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform')) #filter(3,3)
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Flatten()) # flatting the output from the intput layer
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))#100 - no of neurons ,classification layer 
model.add(Dense(10, activation='softmax'))



# In[21]:


model.summary() #32 times filter convaluted


# In[25]:


model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"]) #sgd gradient descent algo


# In[26]:


model_history = model.fit(x_train,y_train,epochs=3,validation_data=(x_valid,y_valid),batch_size=20) # batch size 20 img


# In[ ]:


# plot loss
print(model_history.history.keys())
history_dict= model_history.history
plt.subplot(2, 1, 1)
plt.title('Cross Entropy Loss')
plt.plot(history_dict['loss'], color='blue', label='train')
plt.plot(history_dict['val_loss'], color='orange', label='test')
# plot accuracy
plt.subplot(2, 1, 2)
plt.title('Classification Accuracy')
plt.plot(history_dict['accuracy'], color='blue', label='train')
plt.plot(history_dict['val_accuracy'], color='orange', label='test')
plt.show()


# In[ ]:


model.evaluate(x_test, y_test)


# In[ ]:


from sklearn.model_selection import KFold
kfold = KFold(5, shuffle=True, random_state=1)
	# enumerate splits
for train_ix, test_ix in kfold.split(x_train):
  #define model
  model = Sequential()
  model.add(Conv2D(32,(3, 3),activation='relu',kernel_initializer='he_uniform',input_shape=(28, 28, 1)))
  model.add(MaxPooling2D((2, 2)))
  model.add(Flatten())
  model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
  model.add(Dense(10, activation='softmax'))
  model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])
	# select rows for train and test
  trainX, trainY, testX, testY = x_train[train_ix], y_train[train_ix], x_train[test_ix], y_train[test_ix]
	# fit model
  history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
	# evaluate model
  _, acc = model.evaluate(testX, testY, verbose=0)
  print('> %.3f' % (acc * 100.0))
	
		

