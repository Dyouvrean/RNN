import tensorflow
import pandas as pd
import gensim.downloader as api
import numpy as np
import jieba
from gensim.models import Word2Vec
import torch.nn as nn
import torch
from keras.utils import to_categorical

from model import DOC_RNN
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from keras import regularizers
from keras import layers
from keras import losses
from keras import preprocessing
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# 读取并预处理数据
dev_set = pd.read_csv("./dev.tsv",sep='\t',names=['text','category','id'])
train_set = pd.read_csv("./train.tsv",sep='\t',names=['text','category','id'])
test_set= pd.read_csv("./test.tsv",sep='\t',names=['text','category','id'])

train_set['text'] = train_set.text.apply(lambda x: " ".join(jieba.cut(x)))
dev_set['text']= dev_set.text.apply(lambda x: " ".join(jieba.cut(x)))
test_set['text']= test_set.text.apply(lambda x: " ".join(jieba.cut(x)))
print(train_set['category'].value_counts())

num_words = 20000
tokenizer = Tokenizer(num_words=num_words,oov_token="unk")
tokenizer.fit_on_texts(train_set['text'].tolist())

X_train, X_valid, y_train, y_valid = train_test_split(train_set['text'].tolist(),\
                                                      train_set['category'].tolist(),\
                                                      test_size=0.1,\
                                                      stratify = train_set['category'].tolist(),\
                                                      random_state=0)


print('Train data len:'+str(len(X_train)))
print('Class distribution'+str(Counter(y_train)))
print('Valid data len:'+str(len(X_valid)))
print('Class distribution'+ str(Counter(y_valid)))


x_train = np.array( tokenizer.texts_to_sequences(X_train) )
x_valid = np.array( tokenizer.texts_to_sequences(X_valid) )
x_test  = np.array( tokenizer.texts_to_sequences(test_set['text'].tolist() ))

x_train = pad_sequences(x_train, padding='post', maxlen=40)
x_valid = pad_sequences(x_valid, padding='post', maxlen=40)
x_test = pad_sequences(x_test, padding='post', maxlen=40)


le = LabelEncoder()

train_labels = le.fit_transform(y_train)
train_labels = np.asarray( tensorflow.keras.utils.to_categorical(train_labels))
valid_labels = le.transform(y_valid)
valid_labels = np.asarray( tensorflow.keras.utils.to_categorical(valid_labels))
#
test_labels = le.transform(test_set['category'].tolist())
test_labels = np.asarray(tensorflow.keras.utils.to_categorical(test_labels))
list(le.classes_)

train_ds = tensorflow.data.Dataset.from_tensor_slices((x_train,train_labels))
valid_ds = tensorflow.data.Dataset.from_tensor_slices((x_valid,valid_labels))
test_ds = tensorflow.data.Dataset.from_tensor_slices((x_test,test_labels))

train_labels = le.fit_transform(y_train)
train_labels = np.asarray( tensorflow.keras.utils.to_categorical(train_labels))


count =0

for value,label in train_ds:
    count += 1
    print(value,label)
    if count==3:
        break
count =0

for value,label in valid_ds:
    count += 1
    print(value,label)
    if count==3:
        break
for value,label in test_ds:
    count += 1
    print(value,label)
    if count==3:
        break
max_features = 20000
embedding_dim = 64
sequence_length = 40

model = tensorflow.keras.Sequential()
model.add(tensorflow.keras.layers.Embedding(max_features + 1, embedding_dim, input_length=sequence_length, \
                                    embeddings_regularizer=regularizers.l2(0.0005)))

model.add(tensorflow.keras.layers.Conv1D(128, 3, activation='relu', \
                                 kernel_regularizer=regularizers.l2(0.0005), \
                                 bias_regularizer=regularizers.l2(0.0005)))

model.add(tensorflow.keras.layers.GlobalMaxPooling1D())

model.add(tensorflow.keras.layers.Dropout(0.5))
# 存疑？
model.add(tensorflow.keras.layers.Dense(5, activation='sigmoid',\
                                kernel_regularizer=regularizers.l2(0.001),\
                                bias_regularizer=regularizers.l2(0.001), ))

model.summary()


model.compile(loss=tensorflow.keras.losses.CategoricalCrossentropy(from_logits=True), optimizer='Nadam',
              metrics=["CategoricalAccuracy"])

epochs = 100
history = model.fit(train_ds.shuffle(2000).batch(128),
                    epochs= epochs ,
                    validation_data=valid_ds.batch(128),
                    verbose=1)