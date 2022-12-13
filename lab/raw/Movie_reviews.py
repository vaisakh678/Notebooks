#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import libraries
import pandas as pd
import numpy as np
from keras.layers import LSTM, Activation, Dropout, Dense, Input, Embedding

from keras.models import Model
import string
import re
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
from keras_preprocessing.sequence import pad_sequences
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[2]:


#Movie_reviews.ipynb
# imdbds = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/IMDB Dataset.csv")
# glove_ds = open('/content/drive/MyDrive/Colab Notebooks/glove.6B.50d.txt','r').read()


imdbds = pd.read_csv("/Users/vaisakh/programs/Notebooks/dataset/IMDB Dataset.csv")
glove_ds = open('/Users/vaisakh/programs/Notebooks/dataset/glove.6B.50d.txt','r').read()


# In[3]:


imdbds['review'] =imdbds['review'].str.lower() 


# In[4]:


imdbds.head()


# In[5]:


labelencoder = LabelEncoder()
imdbds['sentiment'] = labelencoder.fit_transform(imdbds['sentiment'])


# In[6]:


imdbds.head()


# In[7]:


#Clean Data
import nltk      
# nltk.download('stopwords')
from nltk.corpus import stopwords 
allstopwords=stopwords.words('english')
imdbds['review']=imdbds['review'].apply(lambda x:" ".join(i for i in x.split() if i not in allstopwords))
imdbds['review']


# In[8]:


import re
imdbds['review']=imdbds['review'].apply(lambda x:''.join(re.findall(r'[a-zA-Z+" "]',x)))
imdbds['review']


# In[9]:


review_lst =(imdbds['review'].values).tolist()
Y =imdbds['sentiment'].values


# In[10]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(review_lst, Y, test_size = 0.2, random_state = 0)


# In[11]:


tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(review_lst)

words_to_index = tokenizer.word_index


# In[12]:


# Load glove pretrained model
def read_glove_vector(glove_vec):
  with open(glove_vec, 'r', encoding='UTF-8') as f:
    words = set()
    word_to_vec_map = {}
    for line in f:
      w_line = line.split()
      curr_word = w_line[0]
      word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)
  return word_to_vec_map


word_to_vec_map = read_glove_vector('../dataset/glove.6B.50d.txt')


# In[13]:


maxLen = 150


# In[14]:


word_to_vec_map


# In[15]:


# Generate glove vectors for each of the words in reviews
vocab_len = len(words_to_index)
embed_vector_len = word_to_vec_map['moon'].shape[0]
emb_matrix = np.zeros((vocab_len, embed_vector_len))
for word, index in words_to_index.items():
  embedding_vector = word_to_vec_map.get(word)
  if embedding_vector is not None:
    emb_matrix[index, :] = embedding_vector


# In[22]:


word_to_vec_map['moon'].shape


# In[16]:


#Make the review sentences uniform length
x_train_indices = tokenizer.texts_to_sequences(x_train)

x_train_indices = pad_sequences(x_train_indices, maxlen=maxLen, padding='post')
x_test_indices = tokenizer.texts_to_sequences(x_test)

x_test_indices = pad_sequences(x_test_indices, maxlen=maxLen, padding='post')


# In[23]:





# In[17]:


#Bild the model
from keras.models import Sequential
from keras import layers
model = Sequential()
model.add(layers.Embedding(input_dim=vocab_len, output_dim=embed_vector_len, input_length=maxLen, weights = [emb_matrix], trainable=False))
model.add(layers.LSTM(128, return_sequences=True))
model.add(layers.Dropout(0.2))

model.add(layers.LSTM(128, return_sequences=True))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM(128, return_sequences=True))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()


# In[18]:


print(x_train_indices.shape)
print(y_train.shape)


# In[19]:


history = model.fit(x_train_indices, y_train,
                    epochs=15,
                    verbose=False,
                    validation_data=(x_test_indices, y_test),
                    batch_size=64)
loss, accuracy = model.evaluate(x_train_indices, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x_test_indices, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))


# In[ ]:





# In[ ]:




