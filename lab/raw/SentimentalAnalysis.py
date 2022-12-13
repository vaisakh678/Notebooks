#!/usr/bin/env python
# coding: utf-8

# In[20]:


from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.metrics import accuracy
from keras import layers
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN


# In[21]:


filepath = "/Users/vaisakh/programs/Notebooks/dataset/Labelled Yelp Dataset.csv"
df = pd.read_csv(filepath, names=["Review", "label"], sep="\t")
df


# In[22]:


review = df["Review"].values
y = df["label"].values
review_train, review_test, y_train, y_test = train_test_split(
    review, y, test_size=0.25, random_state=1000)


# In[23]:


vectorizer = CountVectorizer()
vectorizer.fit(review_train)

X_train = vectorizer.transform(review_train)
X_test = vectorizer.transform(review_test)
X_train


# In[24]:


X_train.shape


# In[25]:


# Model1


input_dim = X_train.shape[1]

model1 = Sequential()
model1.add(layers.Dense(10, input_dim=input_dim, activation="relu"))
model1.add(layers.Dense(1, activation="sigmoid"))


# In[26]:


model1.compile(loss="binary_crossentropy",
               optimizer="adam", metrics=["accuracy"])
model1.summary()


# In[30]:


history = model1.fit(X_train, y_train,
                     epochs=2,

                     validation_data=(X_test, y_test),
                     batch_size=10)


# In[ ]:


loss, accuracy = model1.evaluate(X_train, y_train)
print("Training Accuracy : ", format(accuracy))

loss, accuracy = model1.evaluate(X_test, y_test)
print("Test Accuracy : ", format(accuracy))


# In[31]:


tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(review_train)

X_train = tokenizer.texts_to_sequences(review_train)
X_test = tokenizer.texts_to_sequences(review_test)

vocab_size = len(tokenizer.word_index) + 1
print(review_train[500])
print(X_train[500])
print(vocab_size)


# In[32]:


maxlen = 100

X_train = pad_sequences(X_train, padding="post", maxlen=maxlen)
X_test = pad_sequences(X_test, padding="post", maxlen=maxlen)


# In[33]:


# Model 2

embedding_dem = 50

model2 = Sequential()
model2.add(layers.Embedding(input_dim=vocab_size,
                            output_dim=embedding_dem,
                            input_length=maxlen))
model2.add(layers.Flatten())
model2.add(layers.Dense(10,  activation="relu"))
model2.add(layers.Dense(1, activation="sigmoid"))


model2.compile(loss="binary_crossentropy",
               optimizer="adam", metrics=["accuracy"])
model2.summary()


# In[34]:


history = model2.fit(X_train, y_train,
                     epochs=2,
                     validation_data=(X_test, y_test),
                     batch_size=10)


# In[ ]:


loss, accuracy = model2.evaluate(X_train, y_train)
print("Training Accuracy : ", format(accuracy))

loss, accuracy = model2.evaluate(X_test, y_test)
print("Test Accuracy : ", format(accuracy))


# In[ ]:


# Model 3

embedding_dem = 100

model3 = Sequential()
model3.add(layers.Embedding(vocab_size, embedding_dem, input_length=maxlen))
model3.add(layers.Conv1D(32, 3, activation="relu"))
model3.add(layers.GlobalMaxPooling1D())
model3.add(layers.Dense(10,  activation="relu"))
model3.add(layers.Dense(1, activation="sigmoid"))


model3.compile(loss="binary_crossentropy",
               optimizer="adam", metrics=["accuracy"])


# In[ ]:


history = model3.fit(X_train, y_train,
                     epochs=50,
                     verbose=False,
                     validation_data=(X_test, y_test),
                     batch_size=10)


# In[ ]:


loss, accuracy = model3.evaluate(X_train, y_train, verbose=False,)
print("Training Accuracy : ", format(accuracy))

loss, accuracy = model3.evaluate(X_test, y_test, verbose=False,)
print("Test Accuracy : ", format(accuracy))


# In[ ]:


# 32, 3 = 81.59
# 32, 5 = 80
# 32, 7 = 80.40
# 64, 3 = 80.80
# 64, 5 = 80
# 64, 7 = 79.19
# 128, 3 = 79.60
# 128, 5 = 78.39
# 128, 7 = 80.80


# In[ ]:


# Model 4

embedding_dem = 100

model4 = Sequential()
model4.add(layers.Embedding(vocab_size, embedding_dem, input_length=maxlen))
model4.add(layers.SimpleRNN(128, return_sequences=True))

model4.add(layers.SimpleRNN(128))

model4.add(layers.Dense(10, activation="relu"))

model4.add(layers.Dense(1, activation="softmax"))


model4.compile(loss="binary_crossentropy",
               optimizer="adam", metrics=["accuracy"])


# In[ ]:


history = model4.fit(X_train, y_train,
                     epochs=10,
                     verbose=False,
                     validation_data=(X_test, y_test),
                     batch_size=10)


# In[ ]:


loss, accuracy = model4.evaluate(X_train, y_train, verbose=False,)
print("Training Accuracy : ", format(accuracy))

loss, accuracy = model4.evaluate(X_test, y_test, verbose=False,)
print("Test Accuracy : ", format(accuracy))


# In[ ]:


# model5 LSTM


embedding_dem = 100

model5 = Sequential()
model5.add(layers.Embedding(vocab_size, embedding_dem, input_length=maxlen))
model5.add(layers.LSTM(128, return_sequences=True))
model5.add(layers.Dropout(0.2))
model5.add(layers.LSTM(128))

model5.add(layers.Dense(10, activation="relu"))

model5.add(layers.Dense(1, activation="softmax"))

model5.compile(loss="binary_crossentropy",
               optimizer="adam", metrics=["accuracy"])
model5.summary()


# In[ ]:


history = model5.fit(X_train, y_train,
                     epochs=10,

                     validation_data=(X_test, y_test),
                     batch_size=10)


# In[ ]:


loss, accuracy = model5.evaluate(X_train, y_train, verbose=False,)
print("Training Accuracy : ", format(accuracy))

loss, accuracy = model5.evaluate(X_test, y_test, verbose=False,)
print("Test Accuracy : ", format(accuracy))


# In[ ]:
