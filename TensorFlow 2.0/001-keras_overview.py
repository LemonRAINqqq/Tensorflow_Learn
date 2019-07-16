# -*- coding: utf-8 -*-
"""
https://github.com/czy36mengfei/tensorflow2_tutorials_chinese

001--Keras 快速入门


Created on Mon Jul 15 16:22:31 2019

@author: QinLong
"""
# 1.导入tf.keras
import tensorflow as tf 
from tensorflow.keras import layers

print(tf.__version__)
print(tf.keras.__version__)

# 2.最常见的模型类型是层的堆叠：tf.keras.Sequential 模型
model = tf.keras.Sequential()

model.add(layers.Dence(32))
model.add(layers.Dence(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

'''
activation：设置层的激活函数
kernel_regularizer 和 bias_regularizer：应用层权重（核和偏差）的正则化方案
默认都将不使用
kernel_initializer 和 bias_initializer：创建层权重（核和偏差）的初始化方案 
默认为 "Glorot uniform" 初始化器
'''

layers.Dense(32, kernel_initializer=tf.keras.initializers.glorot_normal)
layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.01))

#3. 训练:
#构建好模型后，通过调用 compile 方法配置该模型的学习流程：

model = tf.keras.Sequential()
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
             loss=tf.keras.losses.categorical_crossentropy,
             metrics=[tf.keras.metrics.categorical_accuracy])
# optimizer  loss  metrics

#3.2. Numpy输入数据
import numpy as np

train_x = np.random.random((1000, 72))
train_y = np.random.random((1000, 10))

val_x = np.random.random((200, 72))
val_y = np.random.random((200, 10))

model.fit(train_x, train_y, epochs=10, batch_size=100,
          validation_data=(val_x, val_y))

#3.3. tf.data输入数据
dataset = tf.data.Dataset.from_tensor_slices((train_x,train_y))
dataset = dataset.batch(32)
dataset = dataset.repeat()
#dataset = tf.data.Dataset.from_tensor_slices((train_x,train_y)).batch(32).repeat()
val_dataset = tf.data.Dataset.from_tensor_slices((val_x,val_y)).batch(32).repeat()

model.fit(dataset, epochs=10,steps_per_epoch=30,
          validation_data=val_dataset,validation_steps=3)


#3.4评估与预测 
test_x = np.random.random((1000, 72))
test_y = np.random.random((1000, 10))
model.evaluate(test_x, test_y, batch_size=32)
test_data = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_data = test_data.batch(32).repeat()
model.evaluate(test_data, steps=30)
# predict
result = model.predict(test_x, batch_size=32)
print(result)


#4.构建高级模型
'''
多输入模型，

多输出模型，

具有共享层的模型（同一层被调用多次），

具有非序列数据流的模型（例如，残差连接）。

层实例可调用并返回张量。 输入张量和输出张量用于定义 tf.keras.Model 实例。和 Sequential 模型一样
'''

input_x = tf.keras.Input(shape=(72,))
hidden1 = layers.Dense(32)(input_x)
hidden2 = layers.Dense(32,activation='relu')(hidden1)
pred = layers.Dense(10,activation='softmax')(hidden2)

model = tf.keras.Model(inputs=input_x, outputs=pred)
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

model.fit(train_x,train_y,batch_size=32,epochs=5)

#4.2模型子类化
'''
通过继承 tf.keras.Model 进行子类化并定义您自己的前向传播来构建完全可自定义的模型。
在 init 方法中创建层并将它们设置为类实例的属性。在 call 方法中定义前向传播
'''
class MyModel(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel,self).__init__(name='my_model')
        self.num_classes = num_classes
        self.layer1 = layers.Dense(32,activation='relu')
        self.layer2 = layers.Dense(10,activation='softmax')
        
    def call(self,inputs):
        h1 = self.layer1(inputs)
        out = self.layer2(h1)
        return out
    
    #compute_output_shape：指定在给定输入形状的情况下如何计算层的输出形状。
    def compute_output_shape(self,input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)
    
    model = MyModel(num_classes=10)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    
    model.fit(train_x,train_y,batch_size=32,epochs=10)

#4.3自定义层  https://keras.io/zh/layers/writing-your-own-keras-layers/
'''
通过对 tf.keras.layers.Layer 进行子类化并实现以下方法来创建自定义层：

build：创建层的权重。使用 add_weight 方法添加权重。

call：定义前向传播。

compute_output_shape：指定在给定输入形状的情况下如何计算层的输出形状。 

或者，可以通过实现 get_config 方法和 from_config 类方法序列化层。
'''
class MyLayer(layers.Layer):
    #数**kwargs代表按字典方式继承父类
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], self.output_dim))
        # 为该层创建一个可训练的权重
        self.kernel = self.add_weight(name='kernel1', 
                                      shape=shape,
                                      initializer='uniform', 
                                      trainable=True)
        super(MyLayer, self).build(input_shape)   # 一定要在最后调用它

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)
#       return (input_shape[0], self.output_dim)
        
    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

model = tf.keras.Sequential(
[
    MyLayer(10),
    layers.Activation('softmax')
])


model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
             loss=tf.keras.losses.categorical_crossentropy,
             metrics=['accuracy'])

model.fit(train_x, train_y, batch_size=16, epochs=5)

'''
还可以定义具有多个输入张量和多个输出张量的 Keras 层。 

为此，你应该假设方法 build(input_shape)，call(x) 和 compute_output_shape(input_shape)

的输入输出都是列表
https://keras.io/zh/layers/writing-your-own-keras-layers/
'''

class MyMulLayer(layers.Layer):
    def __init__(self,output_dim,**kwargs):
        self.output_dim = output.dim
        super(MyMulLayer,self).__init__(**kwargs)
        
    def build(self,input_shape):
        assert isinstance(input_shape,list)
        
        shape = tf.TensorShape((input_shape[1],self.output_dim))
        self.kernel=self.add_weight(name='kernel',
                                    shape=shape,
                                    initializer='uniform',
                                    trainable=True)
        super(MyMulLayer,self).build(input_shape)
        
    def call(self,inputs):
        assert isinstance(inputs,list)
        a,b = inputs
        return [tf.matmul(a,self.kernel) + b, tf.reduce_mean(b, axis=-1)]
    # k.mean ->  return tf.reduce_mean(x, axis, keepdims)
    
    def compute_output_shape(self,input_shape):
        assert isinstance(input_shape,list)
        
        shape_a,shape_b = input_shape
        
# tensoflow 2.0 和 keras 建立的区别？ 后面再说。
        
#4.3回调  你可以使用回调函数来查看训练模型的内在状态和统计
        
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='loss'),
    tf.keras.callbacks.TensorBoard(log_dir='.\logs')
]

model.fit(train_x, train_y, batch_size=16, epochs=5,
         callbacks=callbacks, validation_data=(val_x, val_y))

#5保持和恢复
#5.1权重保存

model.save_weights('./weights/model')
model.load_weights('./weights/model')
model.save_weights('./model.h5')
model.load_weights('./model.h5')


#5.2保存网络结构

# 序列化成json
import json
import pprint
json_str = model.to_json()
pprint.pprint(json.loads(json_str))
fresh_model = tf.keras.models.model_from_json(json_str)

# 保持为yaml格式  #需要提前安装pyyaml
yaml_str = model.to_yaml()
print(yaml_str)
fresh_model = tf.keras.models.model_from_yaml(yaml_str)


#5.3保存整个模型

model.save('all_model.h5')
model = tf.keras.models.load_model('all_model.h5')

'''
6.将keras用于Estimator

Estimator API 用于针对分布式环境训练模型。

它适用于一些行业使用场景，例如用大型数据集进行分布式训练并导出模型以用于生产

'''
estimator = tf.keras.estimator.model_to_estimator(model)



















