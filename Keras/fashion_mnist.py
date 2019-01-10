# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 10:20:28 2019

@author: QinLong
"""

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images,train_labels),(test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#检查训练集
plt.figure()
plt.imshow(train_images[0]) #若图片不显示，改为plt.show
plt.colorbar()
plt.grid(False)

#值缩小到 0 到 1 之间,整数转换为浮点数
train_images = train_images / 255.0
test_images = test_images / 255.0
#验证确保数据格式正确无误
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
    
model =keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128,activation=tf.nn.relu),
        keras.layers.Dense(10,activation=tf.nn.softmax)
        ])
    
model.compile(optimizer=tf.train.AdadeltaOptimizer(),
              #loss=tf.keras.losses.sparse_categorical_crossentropy,
              loss='sparse_categorical_crossentropy',
              #metrics=[tf.keras.metrics.categorical_accuracy]
              metrics=['accuracy']
              )    
#train
model.fit(train_images,train_labels,epochs=5) 

#评估准确率
test_loss,test_acc = model.evaluate(test_images,test_labels)
print('Test accuracy:',test_acc)  
    
'''
conclusion:    
    使用tf.keras搭建model步骤： 框架（model.Sequential）--
                               编译（model.compile)--
                               train（model.fit)--   
                               评估（model.evaluate)--
    
    tf.keras.losses.
    tf.keras.metrics.
    
'''                                 
    
    
    
    
    
    
    
    
    
    
