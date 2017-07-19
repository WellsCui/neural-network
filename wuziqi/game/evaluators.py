import random
import game.interfaces as interfaces
import game.wuziqi as wuziqi

import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)


class EvaluateLearningMode(interfaces.ITrainable):
    def __init__(self, board_size, batch_size, learning_rate, side):
        self.side = side
        self.board_size = board_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # Input Layer
        self.mode = tf.placeholder(tf.string)
        self.states = tf.placeholder(tf.int8, shape=[-1, board_size[0], board_size[1], 1])
        self.y = tf.placeholder("float")
        self.reward = 0
        self.lbd = 0.95
        input_layer = tf.reshape(self.states, [-1, board_size[0], board_size[1], 1])
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        # self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        # self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2, pool_size=[2, 2], strides=2)
        pool2_flat = tf.reshape(conv2, [-1, board_size[0] * board_size[1] * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training=self.mode == learn.ModeKeys.TRAIN)
        self.pred = tf.layers.dense(inputs=dropout, units=1)

        # Mean squared error
        loss = tf.reduce_sum(tf.pow(self.pred - self.y, 2)) / (2 * batch_size)
        # Gradient descent
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
        self.train = optimizer.minimize(loss)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def train(self, data):
        x_train, y_train = self.get_train_data(data)
        for i in range(1000):
            self.sess.run(self.train, {self.states: x_train, self.y: y_train, self.mode: learn.ModeKeys.TRAIN})

    def predict(self, data):
        return self.sess.run(self.pred, {self.states: data, self.mode: learn.ModeKeys.EVAL})

    def get_train_data(self, data):
        x_train = data[:-1]
        y_train = self.reward + self.lbd * self.predict(data[1:])
        return x_train, y_train


class WuziqiEvaluator(interfaces.IEvaluator, interfaces.ITrainable):
    def __init__(self, side):
        self.side = side

    def evaluate(self, environment: interfaces.IEnvironment):
        return

    def train(self, data):
        pass
