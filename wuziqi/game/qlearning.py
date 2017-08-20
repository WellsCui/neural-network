import random
import numpy as np
import game.interfaces as interfaces
import game.wuziqi as wuziqi
import game.utils
import os
import tensorflow as tf
from tensorflow.contrib import learn


class WuziqiQValueNet(interfaces.IEvaluator):
    def __init__(self, board_size, learning_rate, lbd):

        self.board_size = board_size
        self.learning_rate = learning_rate
        # Input Layer
        self.mode = tf.placeholder(tf.string)
        # self.states = tf.placeholder(tf.int8, shape=[batch_size, board_size[0], board_size[1], 1])
        self.state_actions = tf.placeholder(tf.float32)
        self.y = tf.placeholder("float")
        self.lbd = lbd
        self.r = 0.05
        self.kernel_size1 = [2, 2]
        self.kernel_size2 = [2, 2]
        self.kernel_size3 = [2, 2]
        self.pool_size = [2, 2]
        self.training_epics = 300

        input_layer = tf.reshape(
            self.state_actions, [-1, board_size[0], board_size[1], 2],
            name="input_layer")
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

        self.pred = tf.layers.dense(
            inputs=dense, units=1)

        # Mean squared error
        # loss = tf.reduce_sum(tf.pow(self.pred - self.y, 2)) / (2 * batch_size)

        self.loss = tf.losses.mean_squared_error(self.y, self.pred)
        # Gradient descent

        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        init = tf.global_variables_initializer()

        self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='QValueNet')

        # print("trainable_variables:", tf.trainable_variables())
        self.sess = tf.Session()
        self.sess.run(init)

    def evaluate(self, state, action):
        return self.sess.run(self.pred, {self.state_actions: self.build_state_action(state, action),
                                         self.mode: learn.ModeKeys.EVAL})

    def build_state_action(self, state, action):
        action_data = np.zeros(self.board_size)
        action_data[action.x, action.y] = action.val
        result = [state, action_data]
        return result

    def apply_gradient(self, current_state, current_action, r, next_state, next_action):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        grads = optimizer.compute_gradients(self.loss)
        updates = optimizer.apply_gradients(grads)

        self.sess.run(updates, {self.state_actions: self.build_state_action(current_state, current_action),
                                self.y: r + self.lbd * self.evaluate(next_state, next_action),
                                self.mode: learn.ModeKeys.TRAIN})

    def train(self, data):
        state_actions, y = self.build_training_data2(data)

        def eval_epic(epic):
            session = data[0]
            state1, action1, _ = session[-2]
            # state2, action2, _ = session[-3]
            print("epic %d: %f" % (epic, self.evaluate(state1, action1)))
            self.evaluate(state1, action1)
        loss = 0
        for i in range(self.training_epics):
            loss, _ = self.sess.run([self.loss, self.optimizer], {self.state_actions: state_actions,
                                                                  self.y: y,
                                                                  self.mode: learn.ModeKeys.TRAIN})
            if i == 0:
                print("qvalue losses:", loss)
            if (i+1) % 50 == 0:
                eval_epic(i)
        print("qvalue losses:", loss)

    def build_training_data(self, data):
        inputs = []
        y = []
        for session in data:
            for index in range(len(session) - 1):
                state, action, reward = session[index]
                next_state, next_action, next_reward = session[index + 1]
                inputs.append(self.build_state_action(state, action))

                if reward == 1:
                    y.append(1)
                else:
                    y.append(reward + self.lbd * self.evaluate(next_state, next_action))
        return inputs, y

    def build_training_data2(self, data):
        inputs = []
        y = []
        for session in data:
            steps = len(session)
            if steps < 2:
                continue
            end_state, end_action, end_reward = session[-1]
            end_value = 0
            if end_action.val != 0:
                end_value = self.evaluate(end_state, end_action)

            for index in range(steps - 2, 0, -1):
                state, action, reward = session[index]
                inputs.append(self.build_state_action(state, action))
                end_value = reward + self.lbd * end_value
                y.append(end_value)
        return inputs, y

    def save(self, save_path):
        saver = tf.train.Saver()
        save_dir = save_path + "/qvalue_ckpts"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        return saver.save(self.sess, save_dir + "/qvalue_ckpts")

    def restore(self, save_path):
        saver = tf.train.Saver()
        return saver.restore(self.sess, save_path + "/qvalue_ckpts")


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
        self.kernel_size = [5, 5]
        self.pool_size = [2, 2]
        self.training_epics = 50

        input_layer = tf.reshape(
            self.state, [-1, board_size[0], board_size[1], 1], name="policy_input_layer")

        conv1 = tf.layers.conv2d(
            # name="conv1",
            inputs=input_layer,
            filters=16,
            kernel_size=self.kernel_size,
            padding="same",
            activation=tf.nn.relu)

        conv2 = tf.layers.conv2d(
            # name="policy_conv1",
            inputs=conv1,
            filters=32,
            kernel_size=self.kernel_size,
            padding="same",
            activation=tf.nn.relu)
        # self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=[2, 2], strides=2)
        conv3 = tf.layers.conv2d(
            # name="policy_conv2",
            inputs=conv2,
            filters=64,
            kernel_size=self.kernel_size,
            # padding="same",
            activation=tf.nn.relu)

        pool3 = tf.layers.max_pooling2d(
            # name="policy_pool2",
            inputs=conv3, pool_size=self.pool_size, strides=1)

        flat_size = (board_size[0] - (self.kernel_size[0] - 1) - (self.pool_size[0] - 1)) * \
                    (board_size[1] - (self.kernel_size[1] - 1) - (self.pool_size[1] - 1)) * 64
        pool_flat = tf.reshape(pool3, [-1, flat_size])
        dense = tf.layers.dense(
            # name="policy_dense",
            inputs=pool_flat, units=2048, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
            # name="policy_dropout",
            inputs=dense, rate=0.1, training=self.mode == learn.ModeKeys.TRAIN)
        self.pred = tf.layers.dense(
            # name="policy_pred",
            inputs=dropout, units=board_size[0] * board_size[1])
        # self.pred = tf.layers.dense(name="pred", inputs=dense, units=board_size[0] * board_size[1])
        # self.action = tf.layers.dense(name="pred", inputs=dropout, units=1)

        # Mean squared error
        # loss = tf.reduce_sum(tf.pow(self.pred - self.y, 2)) / (2 * batch_size)

        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.pred))
        # Gradient descent

        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        init = tf.global_variables_initializer()

        # print("trainable_variables:", tf.trainable_variables())
        self.sess = tf.Session()
        self.sess.run(init)

    def suggest(self, state, side, count):
        pred = self.sess.run(self.pred, {self.state: state,
                                         self.mode: learn.ModeKeys.EVAL})
        reshaped_pred = np.reshape(pred, self.board_size)
        filled_pos = np.where(state != 0)

        actions = []
        reshaped_pred[filled_pos] = -1
        # legitimate_actions = np.where(reshaped_pred != -1)

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

    def train(self, data):
        states = [s for s, _ in data]
        state_shape = np.shape(states)
        y = np.zeros((state_shape[0], self.board_size[0] * self.board_size[1]))

        for i in range(state_shape[0]):
            action = data[i][1]
            out = np.zeros((self.board_size[0], self.board_size[1]))
            out[action.x, action.y] = 1
            y[i, :] = np.reshape(out, (self.board_size[0] * self.board_size[1]))

        loss = 0
        for i in range(self.training_epics):
            loss, _ = self.sess.run([self.loss, self.optimizer], {self.state: states,
                                                                  self.y: y,
                                                                  self.mode: learn.ModeKeys.TRAIN})
            if i == 0:
                print("policy losses:", loss)
        print("policy losses:", loss)

    def save(self, save_path):
        saver = tf.train.Saver()
        save_dir = save_path + "/policy_ckpts"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        return saver.save(self.sess, save_dir)

    def restore(self, save_path):
        saver = tf.train.Saver()
        return saver.restore(self.sess, save_path + "/policy_ckpts")
