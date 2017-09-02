import numpy as np

import os
import tensorflow as tf
from tensorflow.contrib import learn
import tables
import game.interfaces as interfaces
import game.utils


class WuziqiQValueNet(interfaces.IActionEvaluator):
    def __init__(self, name, board_size, learning_rate, lbd):
        self.name = name
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
        self.training_data_dir = '/tmp/player1/data'
        self.maximum_training_size = 2000

        input_layer = tf.reshape(
            self.state_actions, [-1, board_size[0], board_size[1], 2],
            name=name + "value_net_input_layer")
        conv1 = tf.layers.conv2d(
            name=name + "value_net_conv1",
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
            name=name + "value_net_conv2",
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
            name=name + "value_net_conv3",
            inputs=conv2,
            filters=128,
            kernel_size=self.kernel_size1,
            # padding="same",
            activation=tf.nn.relu)

        pool3 = tf.layers.max_pooling2d(
            name=name + "value_net_pool3",
            inputs=conv3,
            pool_size=self.pool_size,
            strides=1)

        conv4 = tf.layers.conv2d(
            name=name + "value_net_conv4",
            inputs=pool3,
            filters=256,
            kernel_size=self.kernel_size2,
            # padding="same",
            activation=tf.nn.relu)

        conv5 = tf.layers.conv2d(
            name=name + "value_net_conv5",
            inputs=conv4,
            filters=256,
            kernel_size=self.kernel_size2,
            # padding="same",
            activation=tf.nn.relu)

        pool5 = tf.layers.max_pooling2d(
            name=name + "value_net_pool5",
            inputs=conv5,
            pool_size=self.pool_size,
            strides=1)

        conv6 = tf.layers.conv2d(
            name=name + "value_net_conv6",
            inputs=pool5,
            filters=512,
            kernel_size=self.kernel_size3,
            # padding="same",
            activation=tf.nn.relu)

        conv7 = tf.layers.conv2d(
            name=name + "value_net_conv7",
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
        pool_flat = tf.reshape(conv7, [-1, flat_size], name=name + "value_net_pool_flat")
        dropout = tf.layers.dropout(
            name=name + "value_net_dropout",
            inputs=pool_flat, rate=0.7, training=self.mode == learn.ModeKeys.TRAIN)
        dense = tf.layers.dense(
            inputs=dropout, units=2048, activation=tf.nn.relu, name=name + "value_net_dense")

        self.pred = tf.layers.dense(
            inputs=dense, units=1, name=name + "value_net_pred")

        # Mean squared error
        # loss = tf.reduce_sum(tf.pow(self.pred - self.y, 2)) / (2 * batch_size)

        self.loss = tf.losses.mean_squared_error(self.y, self.pred)
        # Gradient descent

        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        # self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='QValueNet')

        # print("trainable_variables:", tf.trainable_variables())
        self.saver = tf.train.Saver()
        self.optimizer = tf.train.AdamOptimizer(learning_rate, name="ValueNet_Optimizer").minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

    def evaluate(self, state, action):
        return self.sess.run(self.pred, {self.state_actions: self.build_state_action(state, action),
                                         self.mode: learn.ModeKeys.EVAL})[0, 0]

    def batch_evaluate(self, state_actions):
        inputs = [self.build_state_action(state, action) for state, action in state_actions]

        return self.sess.run(self.pred, {self.state_actions: inputs,
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
        data_length = self.cached_training_data[1].shape[0]
        if data_length > self.maximum_training_size:
            self.cached_training_data = [
                self.cached_training_data[0][data_length - self.maximum_training_size:],
                self.cached_training_data[1][data_length - self.maximum_training_size:]]
        return self.cached_training_data

    def recall_training_data(self, state_actions, y):
        predicted_y = self.lbd * self.sess.run(self.pred, {self.state_actions: state_actions,
                                                           self.mode: learn.ModeKeys.EVAL})

        index = np.where((y - predicted_y) > (y * (1 - self.lbd)/2))[0]
        print("recall records: %s in %s" % (index.shape[0], y.shape[0]))
        self.cached_training_data = [state_actions[index], y[index]]

    def train(self, learning_rate, data):
        new_training_data = self.build_td_training_data(data)
        self.save_training_data(new_training_data)
        train_data = self.merge_with_cached_training_data(new_training_data)

        if train_data is None or train_data[1].shape[0] <= self.minimum_training_size:
            return 0

        state_actions, y = train_data
        return self.train_with_raw_data(state_actions, y, learning_rate)

    def train_with_raw_data(self, state_actions, y, learning_rate):
        print("Value-Net learning rate: %f train_size: %d" % (learning_rate, state_actions.shape[0]))

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
        # if loss < 0.00005:
        #     return loss

        # eval_epic(-1, loss)

        # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
        for i in range(self.training_epics):
            loss, _ = self.sess.run([self.loss, self.optimizer],
                                    {self.state_actions: state_actions,
                                     self.y: y,
                                     self.mode: learn.ModeKeys.TRAIN})
            if (i + 1) % 50 == 0 or i == 0:
                eval_epic(i, loss)

        # self.recall_training_data(state_actions, y)

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
        save_file = save_path + "/qvalue_ckpts"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return self.saver.save(self.sess, save_file)

    def restore(self, save_path):
        if os.path.isfile(save_path + "/qvalue_ckpts.meta"):
            return self.saver.restore(self.sess, save_path + "/qvalue_ckpts")

    def save_training_data(self, train_data):
        train_file = self.training_data_dir + "/value_net_train.h5"
        if os.path.isfile(train_file):
            f = tables.open_file(train_file, mode='a')
            f.root.train_input.append(train_data[0])
            f.root.train_output.append(train_data[1])
        else:
            f = tables.open_file(train_file, mode='w')
            game.utils.create_earray(f, 'train_input', train_data[0])
            game.utils.create_earray(f, 'train_output', train_data[1])
        f.close()

    def train_with_file(self):
        train_file = self.training_data_dir + "/value_net_train.h5"
        if os.path.isfile(train_file):
            f = tables.open_file(train_file)
            inputs = np.array([x for x in f.root.train_input.iterrows()])
            y = np.array([x for x in f.root.train_output.iterrows()])
            f.close()
            return self.train_with_raw_data(inputs, y, self.learning_rate)
        else:
            return None

