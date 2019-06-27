#-*- encoding: utf-8 -*-

'''
引用URL :  https://blog.codingecho.com/2018/03/25/lstm%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%A6%E3%83%86%E3%82%AD%E3%82%B9%E3%83%88%E3%81%AE%E5%A4%9A%E3%82%AF%E3%83%A9%E3%82%B9%E5%88%86%E9%A1%9E%E3%82%92%E3%81%99%E3%82%8B/
'''

import json
import numpy as np
import csv
import re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
from keras.layers import LSTM, Bidirectional

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

def tokenize(text):
    return text.split(" ")


labels_train = []
samples = []
with open("./AllArgumentSetJumanpp.csv", 'r', encoding="utf-8") as tsv:
    tsv = csv.reader(tsv, delimiter='\t')
    for row in tsv:
        text   = row[0]
        rele   = row[1]
        fact   = row[2]
        stance = row[3]

        if re.match('\d{1}',rele):
            if ( rele == str(1) or rele == str(0)):
                samples.append(text)
                labels_train.append(int(float(rele)))
texts_train = [tokenize(a) for a in samples]

labels_test = []
samples = []
with open("./Tpc&UTRtEST.csv", 'r', encoding="utf-8") as tsv:
    tsv = csv.reader(tsv, delimiter=',')
    for row in tsv:
        text   = row[1]
        rele   = row[0]
#        fact   = row[2]
#        stance = row[3]

        if re.match('\d{1}',rele):
            if ( rele == str(1) or rele == str(0)):
                samples.append(text)
                labels_test.append(int(float(rele)))
texts_test = [tokenize(a) for a in samples]

# print("len(texts):",len(texts_train))

def restore(list,num):
    newList = []
    for index in num:
        newList.append(list[index])
    
    return newList

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from sklearn.model_selection import StratifiedKFold
import numpy

# number for CV
fold_num = 5

seed = 0

num_classes = 2
batch_size  = 32
epochs = 3

max_words = 64

maxlen = 64
midllelen = 32

vocabsize= 10208

# define X-fold cross validation
cvscores = []


'''
# create model
model = Sequential()
model.add(Dense(midllelen, activation='relu', input_shape=(maxlen,)))
model.add(Dropout(0.2))
#model.add(Dense(midllelen, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
optimizer=RMSprop(),
metrics=['accuracy'])
'''

model = Sequential()
model.add(Embedding(input_dim=vocabsize, output_dim=32, input_length=maxlen))
model.add(Bidirectional(LSTM(32)))
#model.add(LSTM(32))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# model.summary()


tokenizer = Tokenizer(num_words=vocabsize)
tokenizer.fit_on_texts(texts_train)

sequences_train = tokenizer.texts_to_sequences(texts_train)


# print(tokenizer.word_index)

# print(sequences_train)
seq_pad_train = pad_sequences(sequences_train,maxlen=maxlen)
# seq_pad_train = tokenizer.texts_to_matrix(texts_train)
# print(len(seq_pad_train[0]))

# print(seq_pad_train)

# Fit the model
model.fit(seq_pad_train, keras.utils.to_categorical(labels_train, num_classes),
          batch_size=batch_size,
          epochs=epochs,
          verbose=0)
    

sequences_test = tokenizer.texts_to_sequences(texts_test)
seq_pad_test = pad_sequences(sequences_test, maxlen=maxlen)
# seq_pad_test = tokenizer.texts_to_matrix(texts_test)

# Evaluate
scores = model.evaluate(seq_pad_test, keras.utils.to_categorical(labels_test, num_classes), verbose=0)

predict_models = model.predict(seq_pad_test, verbose=0)

i = 0
pred_label_int = []
for prob in predict_models:
    # if prob[0] > prob[1]:
    #     print(prob, labels_test[i])
    pred_label_int.append(prob.tolist().index(max(prob.tolist())))
    #    i += 1

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# print(len(predict_models),len(labels_test))

# print(confusion_matrix(labels_test,pred_label_int))

wf = open("BiLSTM.csv","w") 
wf.write('label,A,B\n')
for i in range(len(predict_models)):
    wf.write(str(labels_test[i]) + "," + str(1) + "," + str(pred_label_int[i]) + "\n")

#          cvscores.append(scores[1] * 100)
#

#
#print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
#
#


'''
    maxlen = 1000
    training_samples = len(X_train) # training data 80 : validation data 20
    validation_samples = len(Y_train)
    max_words = 15000

    # word indexを作成
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print("Found {} unique tokens.".format(len(word_index)))

    data = pad_sequences(sequences, maxlen=maxlen)

    # バイナリの行列に変換
    categorical_labels = to_categorical(labels)
    labels = np.asarray(categorical_labels)

    print("Shape of data tensor:{}".format(data.shape))
    print("Shape of label tensor:{}".format(labels.shape))

    # 行列をランダムにシャッフルする
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    x_train = data[:training_samples]
    y_train = labels[:training_samples]
    x_val = data[training_samples: training_samples + validation_samples]
    y_val = labels[training_samples: training_samples + validation_samples]

    model = Sequential()
    model.add(Embedding(15000, 100, input_length=maxlen))
    model.add(LSTM(32))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

    #history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_val, y_val))
    history = model.fit(x_train, y_train, epochs=1, batch_size=200)#, validation_data=(x_val, y_val))

    # evaluate the model
    scores = model.evaluate(x_val, y_val)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

'''
