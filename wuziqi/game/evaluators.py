import random
import game.interfaces as interfaces
import game.wuziqi as wuziqi

import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)


class WuziqiEvaluator(interfaces.IEvaluator, interfaces.IModel):
    def __init__(self, lbd, board_size, batch_size, learning_rate, side):
        self.side = side
        self.board_size = board_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # Input Layer
        self.mode = tf.placeholder(tf.string)
        # self.states = tf.placeholder(tf.int8, shape=[batch_size, board_size[0], board_size[1], 1])
        self.states = tf.placeholder(tf.float32)
        self.y = tf.placeholder("float")
        self.lbd = lbd
        self.r = 0.05
        self.kernel_size = [3, 3]
        self.pool_size = [2, 2]

        input_layer = tf.reshape(
            self.states, [-1, board_size[0], board_size[1], 1], name="input_layer")
        conv1 = tf.layers.conv2d(
            name="conv1",
            inputs=input_layer,
            filters=32,
            kernel_size=self.kernel_size,
            padding="same",
            activation=tf.nn.relu)
        # self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(
            name="conv2",
            inputs=conv1,
            filters=64,
            kernel_size=self.kernel_size,
            # padding="same",
            activation=tf.nn.relu)

        pool2 = tf.layers.max_pooling2d(name="pool2", inputs=conv2, pool_size=self.pool_size, strides=1)

        flat_size = (board_size[0] - (self.kernel_size[0]-1) - (self.pool_size[0] - 1)) * \
                    (board_size[1] - (self.kernel_size[1]-1) - (self.pool_size[1] - 1)) * 64
        pool2_flat = tf.reshape(pool2, [-1, flat_size])
        dense = tf.layers.dense(name="dense", inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(name="dropout",
            inputs=dense, rate=0.4, training=self.mode == learn.ModeKeys.TRAIN)
        self.pred = tf.layers.dense(name="pred", inputs=dropout, units=1)
        # self.action = tf.layers.dense(name="pred", inputs=dropout, units=1)

        # Mean squared error
        # loss = tf.reduce_sum(tf.pow(self.pred - self.y, 2)) / (2 * batch_size)

        self.loss = tf.losses.mean_squared_error(self.y, self.pred)
        # Gradient descent

        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        init = tf.global_variables_initializer()

        print("trainable_variables:", tf.trainable_variables())
        self.sess = tf.Session()
        self.sess.run(init)

    def train(self, data):
        x_train, y_train = self.build_train_data(data)
        losses = []
        epic = 100
        for i in range(epic):
            loss, _ = self.sess.run([self.loss, self.optimizer],
                                    {self.states: x_train, self.y: y_train, self.mode: learn.ModeKeys.TRAIN})
            if i == 0 or i == epic-1:
                losses.append(loss)

        print("losses:", losses)

    def predict(self, data):
        return self.sess.run(self.pred, {self.states: data, self.mode: learn.ModeKeys.EVAL})

    def build_train_data(self, games):
        x_train = []
        y_train = []
        games_shape = np.shape(games)

        for i in range(games_shape[0]):
            g_x_train, g_y_train = self.build_train_data_td_lambda(games[i])
            if len(x_train) == 0:
                x_train = g_x_train
                y_train = g_y_train
            else:
                x_train = np.vstack((x_train, g_x_train))
                y_train = np.vstack((y_train, g_y_train))
        print("shape of training data:", np.shape(x_train))
        return x_train, y_train

    def build_train_data_td_lambda(self, data):
        x_train = data[:-1]
        train_size = np.shape(x_train)[0]
        y_train = np.zeros((train_size, 1))
        r = wuziqi.WuziqiGame.eval_state(self.board_size, data[-1]) * self.side
        target = r
        for i in range(train_size):
            y_train[i] = (1 - self.lbd) * target
            r = self.lbd * r
            target = target + r

        return x_train, y_train

    def build_train_data_td(self, data):
        x_train = data[:-1]
        train_size = np.shape(x_train)[0]
        y_train = np.zeros((train_size, 1))

        for i in range(train_size):
            reward = wuziqi.WuziqiGame.eval_state(self.board_size, data[-1]) * self.side
            y_train[i] = reward + self.lbd * (self.predict(data[i + 1])[0])

        return x_train, y_train

    def build_train_data_td_eligibility_trace(self, data):
        x_train = data[:-1]
        train_size = np.shape(x_train)[0]
        trace = 0
        y_train = np.zeros((train_size, 1))
        bz = self.board_size[0]*self.board_size[0]
        one = np.ones((1, bz))

        for i in range(train_size):
            trace = self.r * self.lbd * trace + one.dot(np.reshape(x_train[i], (bz, 1)))
            reward = wuziqi.WuziqiGame.eval_state(self.board_size, data[-1]) * self.side
            predicts = self.predict(data[i:i + 2])
            delta = reward + self.lbd * predicts[1] - predicts[0]
            y_train[i] = predicts[0] + delta * trace

        return x_train, y_train

    def evaluate(self, environment: interfaces.IEnvironment):
        return self.predict([environment.get_state()])[0]
