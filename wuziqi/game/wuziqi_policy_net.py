import numpy as np
import game.interfaces as interfaces
import game.wuziqi as wuziqi
import os
import tensorflow as tf
from tensorflow.contrib import learn

class WuziqiPolicyNet(interfaces.IPolicy):
    def __init__(self, board_size, learning_rate, lbd):
        self.board_size = board_size
        self.learning_rate = learning_rate
        # Input Layer
        self.mode = tf.placeholder(tf.string)
        self.state = tf.placeholder(tf.float32)
        self.y = tf.placeholder("float")
        self.lbd = lbd
        self.r = 0.05
        self.kernel_size1 = [2, 2]
        self.kernel_size2 = [2, 2]
        self.kernel_size3 = [2, 2]
        self.pool_size = [2, 2]
        self.training_epics = 75
        self.cached_training_data = None
        self.maximum_training_size = 5000

        input_layer = tf.reshape(
            self.state, [-1, board_size[0], board_size[1], 1], name="policy_input_layer")

        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=96,
            kernel_size=self.kernel_size1,
            # padding="same",
            activation=tf.nn.relu)

        # pool1 = tf.layers.max_pooling2d(
        #     inputs=conv1,
        #     pool_size=self.pool_size,
        #     strides=1)

        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=128,
            kernel_size=self.kernel_size1,
            # padding="same",
            activation=tf.nn.relu)

        # pool2 = tf.layers.max_pooling2d(
        #     inputs=conv2,
        #     pool_size=self.pool_size,
        #     strides=1)

        conv3 = tf.layers.conv2d(
            inputs=conv2,
            filters=128,
            kernel_size=self.kernel_size1,
            # padding="same",
            activation=tf.nn.relu)

        pool3 = tf.layers.max_pooling2d(
            inputs=conv3,
            pool_size=self.pool_size,
            strides=1)

        conv4 = tf.layers.conv2d(
            inputs=pool3,
            filters=256,
            kernel_size=self.kernel_size2,
            # padding="same",
            activation=tf.nn.relu)

        conv5 = tf.layers.conv2d(
            inputs=conv4,
            filters=256,
            kernel_size=self.kernel_size2,
            # padding="same",
            activation=tf.nn.relu)

        pool5 = tf.layers.max_pooling2d(
            inputs=conv5,
            pool_size=self.pool_size,
            strides=1)

        conv6 = tf.layers.conv2d(
            inputs=pool5,
            filters=512,
            kernel_size=self.kernel_size3,
            # padding="same",
            activation=tf.nn.relu)

        conv7 = tf.layers.conv2d(
            inputs=conv6,
            filters=512,
            kernel_size=self.kernel_size3,
            # padding="same",
            activation=tf.nn.relu)

        # w = board_size[0] - ((self.kernel_size1[0] - 1) + (self.kernel_size2[0] - 1)*1 + (self.pool_size[0] - 1)*3)
        # h = board_size[1] - ((self.kernel_size1[1] - 1) + (self.kernel_size2[1] - 1)*1 + (self.pool_size[1] - 1)*3)
        #
        # flat_size = w * h * 512
        flat_size = 2048
        pool_flat = tf.reshape(conv7, [-1, flat_size])
        dropout = tf.layers.dropout(
            inputs=pool_flat, rate=0.1, training=self.mode == learn.ModeKeys.TRAIN)
        dense = tf.layers.dense(
            inputs=dropout, units=2048, activation=tf.nn.relu)

        self.scores = tf.layers.dense(inputs=dense, units=board_size[0] * board_size[1])
        self.predictions = tf.argmax(self.scores, 1, name="predictions")

        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.scores))

        correct_predictions = tf.equal(self.predictions, tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # Gradient descent

        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        init = tf.global_variables_initializer()

        # print("trainable_variables:", tf.trainable_variables())
        self.sess = tf.Session()
        self.sess.run(init)

    def suggest(self, state, side, count):
        pred = self.sess.run(self.scores, {self.state: state,
                                           self.mode: learn.ModeKeys.EVAL})
        reshaped_pred = np.reshape(pred, self.board_size)
        filled_pos = np.where(state != 0)

        actions = []
        reshaped_pred[filled_pos] = -1

        for i in range(count):
            index = np.unravel_index(np.argmax(reshaped_pred), self.board_size)
            actions.append(wuziqi.WuziqiAction(index[0], index[1], side))
            reshaped_pred[index] = -1
        return actions

    def apply_gradient(self, current_state, current_action, q_value):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        gradients = optimizer.compute_gradients(self.loss)

        updates = optimizer.apply_gradients(gradients)
        y = np.zeros((self.board_size[0], self.board_size[1]))
        y[current_action.x, current_action.y] = q_value
        y = np.reshape(y, (1, self.board_size[0] * self.board_size[1]))

        self.sess.run(updates, {self.state: current_state,
                                self.y: y,
                                self.mode: learn.ModeKeys.TRAIN})

    def merge_with_cached_training_data(self, training_data):
        if self.cached_training_data is None:
            self.cached_training_data = training_data
        else:
            self.cached_training_data = [
                np.vstack((self.cached_training_data[0], training_data[0])),
                np.vstack((self.cached_training_data[1], training_data[1]))]
        data_length = self.cached_training_data[1].shape[0]
        if data_length > self.maximum_training_size:
            self.cached_training_data = [
                self.cached_training_data[0][data_length - self.maximum_training_size:],
                self.cached_training_data[1][data_length - self.maximum_training_size:]]
        return self.cached_training_data

    def train(self, learning_rate, data):

        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
        states = np.array([s for s, _ in data])
        state_shape = np.shape(states)
        y = np.zeros((state_shape[0], self.board_size[0], self.board_size[1]))

        for i in range(state_shape[0]):
            action = data[i][1]
            y[i, action.y, action.x] = 1.0

        y = y.reshape((state_shape[0], self.board_size[0]*self.board_size[1]))

        states, y = self.merge_with_cached_training_data([states, y])

        print("Policy-Net learning rate: %f training size %s" % (learning_rate, y.shape))

        for i in range(self.training_epics):
            accuracy, _ = self.sess.run([self.accuracy, optimizer], {self.state: states,
                                                             self.y: y,
                                                             self.mode: learn.ModeKeys.TRAIN})
            if (i + 1) % 25 == 0 or i == 0:
                print("epic %d policy accuracy: %f" % (i, accuracy))

    def save(self, save_path):
        saver = tf.train.Saver()
        save_dir = save_path + "/policy_ckpts"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        return saver.save(self.sess, save_dir)

    def restore(self, save_path):
        saver = tf.train.Saver()
        return saver.restore(self.sess, save_path + "/policy_ckpts")
