# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 11:24:14 2019

@author: QinLong
Details: https://www.tensorflow.org/tutorials/keras/basic_text_classification?hl=zh-cn

 IMDB 数据集，其中包含来自互联网电影数据库的 50000 条影评文本。
 我们将这些影评拆分为训练集（25000 条影评）和测试集（25000 条影评）。
 训练集和测试集之间达成了平衡，意味着它们包含相同数量的正面和负面影评。
 
 该数据集已经过预处理：每个样本都是一个整数数组，表示影评中的字词。
 每个标签都是整数值 0 或 1，其中 0 表示负面影评，1 表示正面影评。
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

imdb = keras.datasets.imdb
#参数 num_words=10000 会保留训练数据中出现频次在前 10000位的字词。为确保数据规模处于可管理的水平，罕见字词将被舍弃。
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

#影评文本已转换为整数，其中每个整数都表示字典中的一个特定字词。第一条影评如下所示：
#print(train_data[0])

#将整数转换回字词
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index() #value 从1 开始计数

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index['<PAD>'],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16)) #生成的维度为：(batch, sequence, embedding)
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              #binary_crossentropy 更适合处理概率问题，它可测量概率分布之间的“差距”，在本例中则为实际分布和预测之间的“差距”。
              loss='binary_crossentropy', 
              metrics=['accuracy'])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

results = model.evaluate(test_data, test_labels)
print(results)
#该对象包含一个字典，其中包括训练期间发生的所有情况
history_dict = history.history
#history_dict.keys()


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
