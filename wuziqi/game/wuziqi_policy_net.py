import numpy as np
import os
import tables
import tensorflow as tf
from tensorflow.contrib import learn

import game.interfaces as interfaces
import game.wuziqi as wuziqi
import game.utils

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
        self.kernel_size2 = [3, 3]
        self.kernel_size3 = [3, 3]
        self.pool_size = [2, 2]
        self.training_epics = 100
        self.cached_training_data = None
        self.minimum_training_size = 1000
        self.maximum_training_size = 2000
        self.training_data_dir = 'data'

        input_layer = tf.reshape(
            self.state, [-1, board_size[0], board_size[1], 1], name=name+"policy_input_layer")

        conv1 = tf.layers.conv2d(
            name=name + "policy_conv1",
            inputs=input_layer,
            filters=96,
            kernel_size=self.kernel_size1,
            # padding="same",
            activation=tf.nn.relu)

        conv2 = tf.layers.conv2d(
            name=name + "policy_conv2",
            inputs=conv1,
            filters=96,
            kernel_size=self.kernel_size1,
            # padding="same",
            activation=tf.nn.relu)

        pool2 = tf.layers.max_pooling2d(
            name=name + "policy_pool2",
            inputs=conv2,
            pool_size=self.pool_size,
            strides=1)

        conv3 = tf.layers.conv2d(
            name=name + "policy_conv3",
            inputs=pool2,
            filters=128,
            kernel_size=self.kernel_size2,
            # padding="same",
            activation=tf.nn.relu)

        conv4 = tf.layers.conv2d(
            name=name + "policy_conv4",
            inputs=conv3,
            filters=128,
            kernel_size=self.kernel_size2,
            # padding="same",
            activation=tf.nn.relu)

        pool4 = tf.layers.max_pooling2d(
            name=name + "policy_pool4",
            inputs=conv4,
            pool_size=self.pool_size,
            strides=1)

        conv5 = tf.layers.conv2d(
            name=name + "policy_conv5",
            inputs=pool4,
            filters=256,
            kernel_size=self.kernel_size3,
            # padding="same",
            activation=tf.nn.relu)

        conv6 = tf.layers.conv2d(
            name=name + "policy_conv6",
            inputs=conv5,
            filters=512,
            kernel_size=self.kernel_size3,
            # padding="same",
            activation=tf.nn.relu)

        pool6 = tf.layers.max_pooling2d(
            name=name + "policy_pool6",
            inputs=conv6,
            pool_size=self.pool_size,
            strides=1)

        # conv7 = tf.layers.conv2d(
        #     name=name + "policy_conv7",
        #     inputs=pool6,
        #     filters=512,
        #     kernel_size=self.kernel_size3,
        #     # padding="same",
        #     activation=tf.nn.relu)

        # w = board_size[0] - ((self.kernel_size1[0] - 1) + (self.kernel_size2[0] - 1)*1 + (self.pool_size[0] - 1)*3)
        # h = board_size[1] - ((self.kernel_size1[1] - 1) + (self.kernel_size2[1] - 1)*1 + (self.pool_size[1] - 1)*3)
        #
        # flat_size = w * h * 512
        flat_size = 2048
        pool_flat = tf.reshape(pool6, [-1, flat_size], name=name + "policy_pool_flat")
        dropout = tf.layers.dropout(
            name=name+"policy_dropout",
            inputs=pool_flat,
            rate=0.8,
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
        self.optimizer = tf.train.AdamOptimizer(learning_rate, name="Policy_Optimizer").minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

    def suggest(self, state, side, count):
        pred = self.sess.run(self.scores, {self.state: state,
                                           self.mode: learn.ModeKeys.EVAL})
        reshaped_pred = np.reshape(pred, self.board_size)
        filled_pos = np.where(state != 0)

        actions = []
        min_val = np.amin(reshaped_pred) -1
        reshaped_pred[filled_pos] = min_val

        def is_pos_available(y, x):
            return state[y, x] == 0

        while count > 0:
            index = np.unravel_index(np.argmax(reshaped_pred), self.board_size)
            if is_pos_available(index[0], index[1]):
                actions.append(wuziqi.WuziqiAction(index[1], index[0],  side))
                count -= 1
            reshaped_pred[index] = min_val
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

        states = np.array([s for s, _ in data])
        state_shape = np.shape(states)
        y = np.zeros((state_shape[0], self.board_size[0], self.board_size[1]), dtype=np.int)

        for i in range(state_shape[0]):
            action = data[i][1]
            y[i, action.y, action.x] = 1

        y = y.reshape((state_shape[0], self.board_size[0]*self.board_size[1]))

        self.save_training_data([states, y])

        states, y = self.merge_with_cached_training_data([states, y])

        if y.shape[0] < self.minimum_training_size:
            return [0, 0, 0]

        return self.train_with_raw_data(states, y)

    def train_with_raw_data(self, states, y, log_epic=50, model_dir=None):
        print("Policy-Net learning rate: %f training size %s" % (self.learning_rate, y.shape))

        accuracy, top_5_accuracy, top_10_accuracy = [0, 0, 0]

        for i in range(self.training_epics):
            accuracy, top_5_accuracy, top_10_accuracy, _ = self.sess.run([self.accuracy,
                                                                          self.top_5_accuracy,
                                                                          self.top_10_accuracy,
                                                                          self.optimizer],
                                                                         {self.state: states,
                                                                          self.y: y,
                                                                          self.mode: learn.ModeKeys.TRAIN})
            if (i + 1) % log_epic == 0 or i == 0:
                print("epic %d policy accuracy: %f top_5_accuracy: %f , top_10_accuracy: %f"
                      % (i, accuracy, top_5_accuracy, top_10_accuracy))
            # if model_dir is not None:
            #     print("Saving policy model...")
            #     self.save(model_dir)
        return [accuracy, top_5_accuracy, top_10_accuracy]

    def save(self, save_path):
        save_file = save_path + "/policy_ckpts"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return self.saver.save(self.sess, save_file)

    def restore(self, save_path):
        return self.saver.restore(self.sess, save_path + "/policy_ckpts")

    def save_training_data(self, training_data):
        train_file = self.training_data_dir + "/policy_train.h5"
        if not os.path.exists(self.training_data_dir):
            os.makedirs(self.training_data_dir)
        if os.path.isfile(train_file):
            f = tables.open_file(train_file, mode='a')
            f.root.train_input.append(training_data[0])
            f.root.train_output.append(training_data[1])
        else:
            f = tables.open_file(train_file, mode='w')
            game.utils.create_earray(f, 'train_input', training_data[0])
            game.utils.create_earray(f, 'train_output', training_data[1])
        f.close()

    def train_with_file(self, model_dir):
        train_file = self.training_data_dir + "/policy_train.h5"
        if os.path.isfile(train_file):
            f = tables.open_file(train_file)
            inputs =np.array([x for x in f.root.train_input.iterrows()])
            y = np.array([x for x in f.root.train_output.iterrows()])
            f.close()
            record_count = y.shape[0]
            print("Training policy net with %d records..." % record_count)
            self.merge_with_cached_training_data([inputs, y])
            return self.train_with_raw_data(inputs, y, 5, model_dir)
        else:
            return None
