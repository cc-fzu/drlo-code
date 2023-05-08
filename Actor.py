# 导入相关依赖包
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout
import json
from math import exp
import numpy

# 常量定义
HIDDEN1_UNITS = 1000 # 第一层神经元
HIDDEN2_UNITS = 800 # 第二层神经元


# 激活函数
def tanh_new(inputs):
    with open('parameter.json') as jconfig:
        config = json.load(jconfig)
    tanh_y = config['tanh_y']
    tanh_x = config['tanh_x']
    return tanh_y*(tf.exp(tanh_x*inputs)-tf.exp(-tanh_x*inputs)) / (tf.exp(tanh_x*inputs) + tf.exp(-tanh_x*inputs))

# Actor类
class Actor:
    def __init__(self, state_dim, action_dim, learning_rate):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(self.learning_rate)


    # 创建Actor模型
    def create_model(self):
        model = tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(HIDDEN1_UNITS, activation='relu'),
            Dense(HIDDEN2_UNITS, activation='relu'),
            Dense(self.action_dim, activation = tanh_new)
        ])
        return model

    # 训练网络
    def train(self, states, q_grads, learning_rate):
        with tf.GradientTape() as tape:
            grads = tape.gradient(self.model(states), self.model.trainable_weights, -q_grads)
        print("learning_rate_A: ", learning_rate)
        tf.keras.optimizers.Adam(learning_rate).apply_gradients(zip(grads, self.model.trainable_weights))






