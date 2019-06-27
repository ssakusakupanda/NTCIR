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

def tokenize(text):
    return text.split(" ")

labels = []
samples = []

with open("./Tpc&UTR_Relevance.csv", 'r', encoding="utf-8") as tsv:
    tsv = csv.reader(tsv, delimiter=',')
    for row in tsv:
        text   = row[1]
        stance = row[0]

        if re.match('\d',stance):
            samples.append(text)
            labels.append(int(float(stance)))
texts = [tokenize(a) for a in samples]

print("len(texts):",len(texts))

maxlen = 1000
training_samples = 8000 # training data 80 : validation data 20
validation_samples = len(texts) - training_samples
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
model.add((LSTM(32)))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

model.fit(x_train, y_train, epochs=2, batch_size=32)#, validation_split=0.2)
scores = model.evaluate(x_val, y_val)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

wf = open("miniTest.txt","w")
predictions = model.predict(x_val)
for i in range(len(predictions)):
    #print(y_val[i][0],np.argmax(predictions[i]))
    if int(y_val[i][0]) != np.argmax(predictions[i]):
        wf.write(str(y_val[i][0]) + " " + str(np.argmax(predictions[i])) + "\n")

'''

print("Testing data.")
#data格納 & labels,texts, forms取得
labels  = []
samples = []
forms   = []
with open("./Tpc&UTRtEST.csv", 'r', encoding="utf-8") as tsv:
    tsv = csv.reader(tsv, delimiter=',')
    for row in tsv:
        text   = row[1].split(" [---] ")[0]
        stance = row[0]
        form   = ",".join(row[::]).split(" [---] ")[::][1]
        try:
            samples.append(text)
            labels.append(stance)
            forms.append(form)
        except:
            print(form)
texts = [tokenize(a) for a in samples]
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print("Found {} unique tokens.".format(len(word_index)))

data = pad_sequences(sequences, maxlen=maxlen)


print("over Testing data.")
wf = open("miniTest.txt","w")
wf_values  = open("TestValues.txt","w")
predictions = np.round(model.predict(data))
for index in range(len(predictions)):
    list = predictions[index]
    contents = forms[index].split(",")
    systemRelevance = str(np.where(list == 1)[0].astype(object))
    RelevanceCntent  = ""
    #wf_values.write(forms[index] + " [Relevance]" + str(systemRelevance) + "\n")
    for content in contents:
        try:
            if "Relevance" in content:
                RelevanceCntent += systemRelevance + ","
            else:
                RelevanceCntent += content.split(":")[1] + ","
        except:
            print(",".join((contents)))
            RelevanceCntent = ""
            # RelevanceCntent = contents
    wf.write("".join(RelevanceCntent) + "\n")

# import matplotlib.pyplot as plt
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(1, len(acc) + 1)

# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()

# plt.figure()

# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()

# plt.show()
'''
