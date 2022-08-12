import tensorflow
import numpy as np
import jieba
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
# from tensorflow.python import keras
from keras import regularizers
# from keras import layers
# from keras import losses
# from keras import preprocessing
from collections import Counter
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
# 读取并预处理数据

train_set = pd.read_csv("./train.tsv", sep='\t', names=['text', 'category', 'id'])
test_set = pd.read_csv("./dev.tsv", sep='\t', names=['text', 'category', 'id'])

train_set['text'] = train_set.text.apply(lambda x: " ".join(jieba.cut(x)))
test_set['text'] = test_set.text.apply(lambda x: " ".join(jieba.cut(x)))


num_words = 20000
tokenizer = Tokenizer(num_words=num_words, oov_token="unk")
tokenizer.fit_on_texts(train_set['text'].tolist())


X_train=train_set['text'].tolist();
X_valid=test_set['text'].tolist();
y_train=train_set['category'].tolist();
y_valid=test_set['category'].tolist();


x_train = np.array(tokenizer.texts_to_sequences(X_train))
x_valid = np.array(tokenizer.texts_to_sequences(X_valid))
x_test = np.array(tokenizer.texts_to_sequences(test_set['text'].tolist()))

x_train = pad_sequences(x_train, padding='post', maxlen=40)
x_valid = pad_sequences(x_valid, padding='post', maxlen=40)
x_test = pad_sequences(x_test, padding='post', maxlen=40)

le = LabelEncoder()

train_labels = le.fit_transform(y_train)
train_labels = np.asarray(tensorflow.keras.utils.to_categorical(train_labels))
valid_labels = le.transform(y_valid)
valid_labels = np.asarray(tensorflow.keras.utils.to_categorical(valid_labels))
#
test_labels = le.transform(test_set['category'].tolist())
test_labels = np.asarray(tensorflow.keras.utils.to_categorical(test_labels))
list(le.classes_)

train_ds = tensorflow.data.Dataset.from_tensor_slices((x_train, train_labels))
valid_ds = tensorflow.data.Dataset.from_tensor_slices((x_valid, valid_labels))
test_ds = tensorflow.data.Dataset.from_tensor_slices((x_test, test_labels))

train_labels = le.fit_transform(y_train)
train_labels = np.asarray(tensorflow.keras.utils.to_categorical(train_labels))


max_features = 20000
embedding_dim = 64
sequence_length = 40

model = tensorflow.keras.Sequential()
model.add(tensorflow.keras.layers.Embedding(max_features + 1, embedding_dim, input_length=sequence_length,
                                            embeddings_regularizer=regularizers.l2(0.0005)))

# model.add(tensorflow.keras.layers.Conv1D(128, 3, activation='relu',
#                                          kernel_regularizer=regularizers.l2(0.0005),
#                                          bias_regularizer=regularizers.l2(0.0005)))

model.add(Bidirectional(LSTM(64)))
model.add(tensorflow.keras.layers.Dropout(0.5))
model.add(tensorflow.keras.layers.Dense(5, activation='sigmoid',
                                        kernel_regularizer=regularizers.l2(0.001),
                                        bias_regularizer=regularizers.l2(0.001), ))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 存疑？


model.compile(loss=tensorflow.keras.losses.CategoricalCrossentropy(from_logits=True), optimizer='Nadam',
              metrics=["CategoricalAccuracy"])
model.summary()
epochs = 100
history = model.fit(test_ds.shuffle(2000).batch(128),
                    epochs=epochs,
                    validation_data=valid_ds.batch(128),
                    verbose=1)

plt.plot(history.history['loss'], label=' training data')
plt.plot(history.history['val_loss'], label='validation data)')
plt.title('Loss for Text Classification')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")

plt.savefig('saved_model/rnn/rnn1.png')
plt.show()

plt.plot(history.history['categorical_accuracy'], label=' (training data)')
plt.plot(history.history['val_categorical_accuracy'], label='CategoricalCrossentropy (validation data)')
plt.title('CategoricalAccuracy for Text Classification')
plt.ylabel('CategoricalAccuracy value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")

plt.savefig('saved_model/rnn/rnn2.png')
plt.show()


model.save('saved_model/rnn')
json_string = tokenizer.to_json()

import json
with open('saved_model/rnn_tokenizer.json', 'w') as outfile:
    json.dump(json_string, outfile)

new_model = tensorflow.keras.models.load_model('saved_model/rnn')
new_model.summary()

with open('saved_model/rnn_tokenizer.json') as json_file:
    json_string = json.load(json_file)

tokenizer1 = tensorflow.keras.preprocessing.text.tokenizer_from_json(json_string)


eval_set = pd.read_csv("./test.tsv",sep='\t',names=['text','category','id'])
eval_set['text']= eval_set.text.apply(lambda x: " ".join(jieba.cut(x)))
x_test  = np.array( tokenizer1.texts_to_sequences(test_set['text'].tolist()))
x_test = pad_sequences(x_test, padding='post', maxlen=40)
predictions = new_model.predict(x_test)
predict_results = predictions.argmax(axis=1)

test_set['pred_sentiment']= predict_results
labels = [0,1,2,3,4]
print(classification_report(test_set['category'].tolist(), test_set['pred_sentiment'].tolist(), labels=labels))

