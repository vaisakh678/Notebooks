#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN
import matplotlib.pyplot as plt


# In[2]:


mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train1=x_train/255.0
x_test1=x_test/255.0


# In[3]:


print(x_train1.shape)


# In[4]:


model=Sequential()
model.add(SimpleRNN(128, input_shape=(28,28),return_sequences=True))
model.add(Dropout(0.2))

model.add(SimpleRNN(128))
model.add(Dropout(0.2))

model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10,activation='softmax'))


model.summary()


# In[5]:


model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])


# In[6]:


model_history = model.fit(x_train1,y_train,epochs=3,validation_data=(x_test1,y_test))


# In[ ]:


scores=model.evaluate(x_test,y_test,verbose=0)
print("Accuracy: %.2f%%" %(scores[1]*100))
# loss, accuracy = model.evaluate(x_test, y_train,verbose=0 )
# print("Training Accuracy : ", format(accuracy))


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


#implementing LSTM

model2=Sequential()
model2.add(LSTM(128, input_shape=(28,28),return_sequences=True))
model2.add(Dropout(0.2))

model2.add(LSTM(128))
model2.add(Dropout(0.2))

model2.add(Dense(32,activation='relu'))
model2.add(Dropout(0.2))

model2.add(Dense(10,activation='softmax'))


model2.summary()


# In[ ]:


model2.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])


# In[ ]:


model2_history = model2.fit(x_train1,y_train,epochs=3,validation_data=(x_test1,y_test))


# In[ ]:


scores=model2.evaluate(x_train1,y_train,verbose=0)
print("Accuracy: %.2f%%" %(scores[1]*100))
scores


# how to load CIFAR10
# implement cnn in this program
# 
