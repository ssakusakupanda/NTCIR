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

def tokenize(text):
    return text.split(" ")


labels = []
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
                labels.append(int(float(rele)))
texts = [tokenize(a) for a in samples]

print("len(texts):",len(texts))

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
batch_size  = 128
epochs = 2

max_words = 128

maxlen = 768
midllelen = 128
# define X-fold cross validation
kfold = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=seed)
cvscores = []

for train, test in kfold.split(texts, labels):
    
    '''
    # create model
    model = Sequential()
    model.add(Dense(midllelen, activation='relu', input_shape=(maxlen,)))
    model.add(Dropout(0.2))
    model.add(Dense(midllelen, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
    optimizer=RMSprop(),
    metrics=['accuracy'])

    '''
    
    model = Sequential()
    model.add(Embedding(150, 100, input_length=maxlen))
    model.add(LSTM(32))
    model.add(Dense(2, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    
        
    texts_train  = restore(texts,train)
    labels_train = restore(labels,train)

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts_train)
    sequences_train = tokenizer.texts_to_sequences(texts_train)

#    print(tokenizer.word_index)

    print(sequences_train)

    seq_pad_train = pad_sequences(sequences_train,maxlen=maxlen)

    # Fit the model
    model.fit(seq_pad_train, keras.utils.to_categorical(labels_train, num_classes),
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1)
                            
    texts_test  = restore(texts,test)
    labels_test = restore(labels,test)
    
    sequences_test = tokenizer.texts_to_sequences(texts_test)

    seq_pad_test = pad_sequences(sequences_test, maxlen=maxlen)
    
    # Evaluate
    scores = model.evaluate(seq_pad_test, keras.utils.to_categorical(labels_test, num_classes), verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))


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
