# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 16:30:03 2019

@author: QinLong

Details: https://www.tensorflow.org/tutorials/keras/overfit_and_underfit?hl=zh-cn

overfitting and underfitting ：两种常见的正则化技术（权重正则化和丢弃）
改进我们的 IMDB 影评分类笔记
"""

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

NUM_WORDS = 10000

(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)

# 多热编码
def multi_hot_sequences(sequences, dimension):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0  # set specific indices of results[i] to 1s
    return results

train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

#plt.plot(train_data[0])

base_model  = keras.Sequential([
        keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(1,  activation=tf.nn.sigmoid)
        ])

#base_model.compile(optimizer=tf.train.AdadeltaOptimizer(),
#                   loss=tf.keras.losses.binary_crossentropy,
#                   metrics=[tf.keras.metrics.binary_crossentropy])
# 这种写法与下面train时 acc loss 不同， why       
base_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy', 'binary_crossentropy'])
base_model.summary()

base_history = base_model.fit(train_data,
                              train_labels,
                              epochs=20,
                              batch_size=512,
                              validation_data=(test_data,test_labels),
                              #verbose: 0, 1 或 2。日志显示模式。 0 = 安静模式, 1 = 进度条, 2 = 每轮一行。
                              verbose=1) 

def plot_history(histories, key='binary_crossentropy'):
  plt.figure(figsize=(16,10))

  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()

  plt.xlim([0,max(history.epoch)])

#网络容量越大，便能够越快对训练数据进行建模（产生较低的训练损失），
#但越容易过拟合（导致训练损失与验证损失之间的差异很大）。
'''
    奥卡姆剃刀定律.
    添加权重正则化:
        L1 正则化，其中所添加的代价与权重系数的绝对值（即所谓的权重“L1 范数”）成正比。
        L2 正则化，其中所添加的代价与权重系数值的平方（即所谓的权重“L2 范数”）成正比。
        L2 正则化在神经网络领域也称为权重衰减。不要因为名称不同而感到困惑：从数学角度来讲，
        权重衰减与 L2 正则化完全相同。

'''
l2_model = keras.models.Sequential([
        keras.layers.Dense(16,kernel_regularizer=keras.regularizers.l2(0.001),
                           activation=tf.nn.relu,input_shape=(NUM_WORDS,)),
        keras.layers.Dense(16,kernel_regularizer=keras.regularizers.l2(0.001),
                           activation=tf.nn.relu),
        keras.layers.Dense(1,activation=tf.nn.sigmoid)
        ])
#sigmoid函数作为神经元的激活函数时，最好使用交叉熵代价函数来替代方差代价函数，以避免训练过程太慢。
l2_model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy','binary_crossentropy'])

l2_history = l2_model.fit(train_data, train_labels,
             epochs=20,
             batch_size=512,
             validation_data=(test_data, test_labels),
             verbose=2)
#l2(0.001) 表示层的权重矩阵中的每个系数都会将 0.001 * weight_coefficient_value**2 添加到网络的总损失中。
#请注意，由于此惩罚仅在训练时添加，此网络在训练时的损失将远高于测试时。
plot_history([('base', base_history),
              ('l2', l2_history)])

#添加丢弃层 
from keras.layers import Dense,Dropout
from keras.models import Sequential

dpt_model = Sequential([
        Dense(16,activation=tf.nn.relu,input_shape=(NUM_WORDS,)),
        Dropout(0.5),
        Dense(16,activation=tf.nn.relu),
        Dropout(0.5),
        Dense(1,activation=tf.nn.sigmoid)
        ])
dpt_model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy','binary_crossentropy'])

dpt_history = dpt_model.fit(train_data, train_labels,
             epochs=20,
             batch_size=512,
             validation_data=(test_data, test_labels),
             verbose=2)

plot_history([('base', base_history),
              ('dropout', dpt_history)])

'''conclusion:
    
防止神经网络出现过拟合的最常见方法：

    获取更多训练数据。
    降低网络容量。
    添加权重正则化。
    添加丢弃层。
还有两个重要的方法在本指南中没有介绍：  数据增强
                                    批次归一化。

'''



