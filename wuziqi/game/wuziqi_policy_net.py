import numpy as np
import game.interfaces as interfaces
import game.wuziqi as wuziqi
import os
import tensorflow as tf
from tensorflow.contrib import learn

class WuziqiPolicyNet(interfaces.IPolicy):
    def __init__(self, name, board_size, learning_rate, lbd):
        self.name = name
        self.board_size = board_size
        self.learning_rate = learning_rate
        # Input Layer
        self.mode = tf.placeholder(tf.string)
        self.state = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.int32, shape=[None, board_size[0] * board_size[1]])
        self.lbd = lbd
        self.r = 0.05
        self.kernel_size1 = [2, 2]
        self.kernel_size2 = [2, 2]
        self.kernel_size3 = [2, 2]
        self.pool_size = [2, 2]
        self.training_epics = 100
        self.cached_training_data = None
        self.maximum_training_size = 2000

        input_layer = tf.reshape(
            self.state, [-1, board_size[0], board_size[1], 1], name=name+"policy_input_layer")

        conv1 = tf.layers.conv2d(
            name=name+"policy_conv1",
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
            name=name+"policy_conv2",
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
            name=name+"policy_conv3",
            inputs=conv2,
            filters=128,
            kernel_size=self.kernel_size1,
            # padding="same",
            activation=tf.nn.relu)

        pool3 = tf.layers.max_pooling2d(
            name=name+"policy_pool3",
            inputs=conv3,
            pool_size=self.pool_size,
            strides=1)

        conv4 = tf.layers.conv2d(
            name=name+"policy_conv4",
            inputs=pool3,
            filters=256,
            kernel_size=self.kernel_size2,
            # padding="same",
            activation=tf.nn.relu)

        conv5 = tf.layers.conv2d(
            name=name+"policy_conv5",
            inputs=conv4,
            filters=256,
            kernel_size=self.kernel_size2,
            # padding="same",
            activation=tf.nn.relu)

        pool5 = tf.layers.max_pooling2d(
            name=name+"policy_pool5",
            inputs=conv5,
            pool_size=self.pool_size,
            strides=1)

        conv6 = tf.layers.conv2d(
            name=name+"policy_conv6",
            inputs=pool5,
            filters=512,
            kernel_size=self.kernel_size3,
            # padding="same",
            activation=tf.nn.relu)

        conv7 = tf.layers.conv2d(
            name=name+"policy_conv7",
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
        pool_flat = tf.reshape(conv7, [-1, flat_size], name=name+"policy_flat")
        dropout = tf.layers.dropout(
            name=name+"policy_dropout",
            inputs=pool_flat,
            rate=0.1,
            training=self.mode == learn.ModeKeys.TRAIN)
        dense = tf.layers.dense(
            name=name+"policy_dense",
            inputs=dropout,
            units=2048,
            activation=tf.nn.relu)

        self.scores = tf.layers.dense(
            name=name+"policy_scores",
            inputs=dense, units=board_size[0] * board_size[1])
        self.predictions = tf.argmax(self.scores, 1, name=name+"predictions")

        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                name=name+"policy_cross_entropy",
                labels=self.y, logits=self.scores),
            name=name+"policy_loss")

        y_index = tf.argmax(self.y, 1, name=name+"policy_y_index" )

        correct_predictions = tf.equal(self.predictions, y_index, name=name+"correct_predictions")
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name=name+"accuracy")
        correct_predictions_in_top_5 = tf.nn.in_top_k(predictions=self.scores, targets=y_index, k=5, name=name+"top_5_predictions")
        correct_predictions_in_top_10 = tf.nn.in_top_k(predictions=self.scores, targets=y_index, k=10, name=name+"top_10_predictions")
        self.top_5_accuracy = tf.reduce_mean(tf.cast(correct_predictions_in_top_5, "float", name=name+"top_5_accuracy"))
        self.top_10_accuracy = tf.reduce_mean(tf.cast(correct_predictions_in_top_10, "float",name=name+"top_10_accuracy"))

        # Gradient descent

        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        # print("trainable_variables:", tf.trainable_variables())
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

    def suggest(self, state, side, count):
        pred = self.sess.run(self.scores, {self.state: state,
                                           self.mode: learn.ModeKeys.EVAL})
        reshaped_pred = np.reshape(pred, self.board_size)
        filled_pos = np.where(state != 0)

        actions = []
        reshaped_pred[filled_pos] = -1

        def is_pos_available(x, y):
            return state[x, y] == 0

        while count > 0:
            index = np.unravel_index(np.argmax(reshaped_pred), self.board_size)
            if is_pos_available(index[0], index[1]):
                actions.append(wuziqi.WuziqiAction(index[0], index[1], side))
                count -= 1
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
        y = np.zeros((state_shape[0], self.board_size[0], self.board_size[1]), dtype=np.int)

        for i in range(state_shape[0]):
            action = data[i][1]
            y[i, action.x, action.y] = 1

        y = y.reshape((state_shape[0], self.board_size[0]*self.board_size[1]))

        states, y = self.merge_with_cached_training_data([states, y])

        print("Policy-Net learning rate: %f training size %s" % (learning_rate, y.shape))

        accuracy, top_5_accuracy, top_10_accuracy = [0, 0, 0]

        for i in range(self.training_epics):
            accuracy, top_5_accuracy, top_10_accuracy, _ = self.sess.run([self.accuracy,
                                                                          self.top_5_accuracy,
                                                                          self.top_10_accuracy,
                                                                          optimizer],
                                                                         {self.state: states,
                                                                          self.y: y,
                                                                          self.mode: learn.ModeKeys.TRAIN})
            if (i + 1) % 25 == 0 or i == 0:
                print("epic %d policy accuracy: %f top_5_accuracy: %f , top_10_accuracy: %f"
                      % (i, accuracy, top_5_accuracy, top_10_accuracy))
        return [accuracy, top_5_accuracy, top_10_accuracy]

    def save(self, save_path):
        save_file = save_path + "/policy_ckpts"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return self.saver.save(self.sess, save_file)

    def restore(self, save_path):
        return self.saver.restore(self.sess, save_path + "/policy_ckpts")
