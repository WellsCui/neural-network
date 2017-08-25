import random
import numpy as np
import game.interfaces as interfaces
import game.wuziqi as wuziqi
import game.utils
import os
import tensorflow as tf
from tensorflow.contrib import learn


class WuziqiQValueNet(interfaces.IActionEvaluator):
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
        self.training_epics = 100
        self.minimum_training_size = 100
        self.cached_training_data = None

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
            inputs=pool_flat, rate=0.5, training=self.mode == learn.ModeKeys.TRAIN)
        dense = tf.layers.dense(
            inputs=dropout, units=2048, activation=tf.nn.relu)

        self.pred = tf.layers.dense(
            inputs=dense, units=1)

        # Mean squared error
        # loss = tf.reduce_sum(tf.pow(self.pred - self.y, 2)) / (2 * batch_size)

        self.loss = tf.losses.mean_squared_error(self.y, self.pred)
        # Gradient descent

        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        init = tf.global_variables_initializer()

        self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='QValueNet')

        # print("trainable_variables:", tf.trainable_variables())
        self.sess = tf.Session()
        self.sess.run(init)

    def evaluate(self, state, action):
        return self.sess.run(self.pred, {self.state_actions: self.build_state_action(state, action),
                                         self.mode: learn.ModeKeys.EVAL})[0, 0]

    def suggest(self, environment, candidate_actions, suggest_count=1):
        state = environment.get_state().copy()
        reversed_state = environment.get_state().copy()
        candidate_count = len(candidate_actions)
        evaluate_data = np.vstack(([self.build_state_action(state, action) for action in candidate_actions],
                                   [self.build_state_action(reversed_state, action) for action in candidate_actions]))

        results = self.sess.run(self.pred, {self.state_actions: evaluate_data,
                                            self.mode: learn.ModeKeys.EVAL})

        results = np.hstack((results[:candidate_count], results[candidate_count:]))

        chosen = []
        for i in range(suggest_count):
            max_index = np.unravel_index(np.argmax(results), results.shape)
            chosen.append(candidate_actions[max_index[0]])
            results[max_index] = -1
        return chosen

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

    def merge_with_cached_training_data(self, training_data):
        if training_data[1].shape[0] == 0:
            return self.cached_training_data
        if self.cached_training_data is None:
            self.cached_training_data = training_data
        else:
            self.cached_training_data = [
                np.vstack((self.cached_training_data[0], training_data[0])),
                np.vstack((self.cached_training_data[1], training_data[1]))]
        return self.cached_training_data

    def recall_training_data(self, state_actions, y):
        predicted_y = self.lbd * self.sess.run(self.pred, {self.state_actions: state_actions,
                                                           self.mode: learn.ModeKeys.EVAL})

        index = np.where((y - predicted_y) > (y * (1 - self.lbd)/2))[0]
        print("recall records: %s in %s" % (index.shape[0], y.shape[0]))
        self.cached_training_data = [state_actions[index], y[index]]

    def train(self, learning_rate, data):
        train_data = self.merge_with_cached_training_data(self.build_td_training_data(data))

        if train_data is None or train_data[1].shape[0] <= self.minimum_training_size:
            return 0

        state_actions, y = train_data

        print("Value-Net learning rate: %f train_size: %d" % (learning_rate, train_data[1].shape[0]))

        def eval_epic(epic, loss):
            print("epic %d: %f " % (epic, loss))
            vals = self.sess.run(self.pred, {self.state_actions: state_actions[0:20],
                                             self.mode: learn.ModeKeys.EVAL})
            result = np.append(vals, np.reshape(y[0:20], vals.shape), axis=1)
            print(result)

        loss = self.sess.run(self.loss,
                                {self.state_actions: state_actions,
                                 self.y: y,
                                 self.mode: learn.ModeKeys.TRAIN})
        if loss < 0.00005:
            return loss

        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
        for i in range(self.training_epics):
            loss, _ = self.sess.run([self.loss, optimizer],
                                    {self.state_actions: state_actions,
                                     self.y: y,
                                     self.mode: learn.ModeKeys.TRAIN})
            if (i + 1) % 25 == 0 or i == 0:
                eval_epic(i, loss)

        self.recall_training_data(state_actions, y)

        return loss

    def build_td_0_training_data(self, data):
        inputs = None
        y = None
        for session in data:
            steps = len(session)
            if steps < 2:
                continue

            session_inputs = np.array([self.build_state_action(state, action) for state, action, _ in session])
            session_y = self.lbd * self.sess.run(self.pred, {self.state_actions: session_inputs[1:],
                                            self.mode: learn.ModeKeys.EVAL})
            for i in range(steps - 1):
                state, action, reward = session[i]
                if reward == 1:
                    session_y[i] = [1]
            if inputs is None:
                inputs = session_inputs[:-1]
            else:
                inputs = np.vstack((inputs, session_inputs[:-1]))
            if y is None:
                y = session_y
            else:
                y = np.vstack((y, session_y))
        return inputs, y

    def build_td_training_data(self, data):
        inputs = []
        y = []
        margin = (1 - self.lbd) / 2
        for session in data:
            steps = len(session)
            if steps < 2:
                continue
            end_state, end_action, end_reward = session[-1]
            end_value = 0
            if end_action.val != 0:
                end_value = self.evaluate(end_state, end_action)
                if end_value - self.lbd > margin:
                    end_value = self.lbd
                    inputs.append(self.build_state_action(end_state, end_action))
                    y.append([end_value])
                elif end_value < margin * 2:
                    continue

            session_inputs = np.array([self.build_state_action(state, action) for state, action, _ in session])
            session_y = self.lbd * self.sess.run(self.pred, {self.state_actions: session_inputs[1:],
                                                             self.mode: learn.ModeKeys.EVAL})

            for index in range(steps - 2, 0, -1):
                state, action, reward = session[index]
                end_value = reward + self.lbd * end_value
                if end_value - session_y[index] > end_value * margin:
                    inputs.append(self.build_state_action(state, action))
                    y.append([end_value])
                elif session_y[index] - self.lbd > self.lbd * margin and reward == 0:
                    inputs.append(self.build_state_action(state, action))
                    y.append([self.lbd])
                # elif end_value - self.lbd > (1 - self.lbd) / 2:
                #     end_value = self.lbd
                #     inputs.append(self.build_state_action(end_state, end_action))
                #     y.append([end_value])
                elif reward == 1 and (session_y[index] - 1) > margin:
                    inputs.append(self.build_state_action(state, action))
                    y.append([1])

        return np.array(inputs), np.array(y)

    def save(self, save_path):
        saver = tf.train.Saver()
        save_dir = save_path + "/qvalue_ckpts"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        return saver.save(self.sess, save_dir + "/qvalue_ckpts")

    def restore(self, save_path):
        saver = tf.train.Saver()
        return saver.restore(self.sess, save_path + "/qvalue_ckpts")

