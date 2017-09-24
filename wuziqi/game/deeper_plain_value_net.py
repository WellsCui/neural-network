import numpy as np
import logging
import os
import tensorflow as tf
from tensorflow.contrib import learn, layers
import tables
import game.interfaces as interfaces
import game.utils


class DeeperPlainValueNet(object):
    def __init__(self, name, board_size, learning_rate, lbd):
        self.name = name
        self.board_size = board_size
        self.learning_rate = learning_rate
        # Input Layer
        self.mode = tf.placeholder(tf.string)
        # self.states = tf.placeholder(tf.int8, shape=[batch_size, board_size[0], board_size[1], 1])
        self.inputs = tf.placeholder(tf.float32)
        self.y = tf.placeholder("float")
        self.lbd = lbd
        self.r = 0.05
        self.kernel_size = [2, 2]

        self.pool_size = [2, 2]
        self.training_epics = 10
        self.minimum_training_size = 20
        self.maximum_training_size = 500000
        self.cached_training_data = None
        self.training_data_dir = 'data'
        self.model_path = 'model'
        self.logger = logging.root

        input_layer = tf.reshape(
            self.inputs, [-1, board_size[0], board_size[1], 1],
            name=name + "value_net_input_layer")

        bn1 = tf.layers.batch_normalization(input_layer, training=self.mode == learn.ModeKeys.TRAIN, name="value_net_bn1")

        conv1 = tf.layers.conv2d(
            name=name + "value_net_conv1",
            inputs=bn1,
            filters=81,
            kernel_size=self.kernel_size,
            # padding="same",
            activation=tf.nn.relu)

        bn2 = tf.layers.batch_normalization(conv1, training=self.mode == learn.ModeKeys.TRAIN, name="value_net_bn2")

        conv2 = tf.layers.conv2d(
            name=name + "value_net_conv2",
            inputs=bn2,
            filters=81,
            kernel_size=self.kernel_size,
            # padding="same",
            activation=tf.nn.relu)

        bn3 = tf.layers.batch_normalization(conv2, training=self.mode == learn.ModeKeys.TRAIN, name="value_net_bn3")

        conv3 = tf.layers.conv2d(
            name=name + "value_net_conv3",
            inputs=bn3,
            filters=81,
            kernel_size=self.kernel_size,
            # padding="same",
            activation=tf.nn.relu)

        pool3 = tf.layers.max_pooling2d(
            name=name + "value_net_pool3",
            inputs=conv3,
            pool_size=self.pool_size,
            strides=1)

        bn4 = tf.layers.batch_normalization(pool3, training=self.mode == learn.ModeKeys.TRAIN, name="value_net_bn4")

        conv4 = tf.layers.conv2d(
            name=name + "value_net_conv4",
            inputs=bn4,
            filters=128,
            kernel_size=self.kernel_size,
            # padding="same",
            activation=tf.nn.relu)

        bn5 = tf.layers.batch_normalization(conv4, training=self.mode == learn.ModeKeys.TRAIN, name="value_net_bn5")

        conv5 = tf.layers.conv2d(
            name=name + "value_net_conv5",
            inputs=bn5,
            filters=128,
            kernel_size=self.kernel_size,
            # padding="same",
            activation=tf.nn.relu)

        bn6 = tf.layers.batch_normalization(conv5, training=self.mode == learn.ModeKeys.TRAIN, name="value_net_bn6")

        conv6 = tf.layers.conv2d(
            name=name + "value_net_conv6",
            inputs=bn6,
            filters=128,
            kernel_size=self.kernel_size,
            # padding="same",
            activation=tf.nn.relu)

        pool6 = tf.layers.max_pooling2d(
            name=name + "value_net_pool6",
            inputs=conv6,
            pool_size=self.pool_size,
            strides=1)

        bn7 = tf.layers.batch_normalization(pool6, training=self.mode == learn.ModeKeys.TRAIN, name="value_net_bn7")

        conv7 = tf.layers.conv2d(
            name=name + "value_net_conv7",
            inputs=bn7,
            filters=256,
            kernel_size=[3, 3],
            # padding="same",
            activation=tf.nn.relu)

        bn8 = tf.layers.batch_normalization(conv7, training=self.mode == learn.ModeKeys.TRAIN, name="value_net_bn8")

        conv8 = tf.layers.conv2d(
            name=name + "value_net_conv8",
            inputs=bn8,
            filters=512,
            kernel_size=[3, 3],
            # padding="same",
            activation=tf.nn.relu)

        pool8 = tf.layers.max_pooling2d(
            name=name + "value_net_pool8",
            inputs=conv8,
            pool_size=self.pool_size,
            strides=1)

        flat_size = 2048
        pool_flat = tf.reshape(pool8, [-1, flat_size], name=name + "value_net_pool_flat")
        dropout = tf.layers.dropout(
            name=name + "value_net_dropout",
            inputs=pool_flat, rate=0.2, training=self.mode == learn.ModeKeys.TRAIN)
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
        self.optimizer = tf.train.AdamOptimizer(learning_rate, name="value_net_Optimizer").minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

    def evaluate(self, state, action):
        new_state = state.copy()
        new_state[action.y, action.x] = action.val
        return self.batch_evaluate(new_state)[0, 0]

    def batch_evaluate(self, states):
        return self.sess.run(self.pred, {self.inputs: states,
                                         self.mode: learn.ModeKeys.EVAL})

    def suggest(self, environment, candidate_actions, suggest_count=1):
        states = []
        candidate_count = len(candidate_actions)

        for a in candidate_actions:
            env = environment.clone()
            env.update(a)
            if env.eval_state() == 1:
                return [a]
            states.append(env.get_state())
            env = environment.reverse().clone()
            env.update(a)
            if env.eval_state() == 1:
                return [a]
            states.append(env.get_state())

        results = self.batch_evaluate(np.array(states))

        if self.logger.level == logging.DEBUG:
            action_string = ['(%d, %d): %f %f' %
                             (candidate_actions[i].x, candidate_actions[i].y, results[i], results[candidate_count+i])
                             for i in range(candidate_count)]
            self.logger.debug(" suggestion candidates: %s", ','.join(action_string))

        chosen = []
        for i in range(suggest_count):
            max_index = np.unravel_index(np.argmax(results), results.shape)
            chosen.append(candidate_actions[max_index[0] % candidate_count])
            results[max_index] = -1

        return chosen

    def merge_with_cached_training_data(self, training_data):
        if training_data[1].shape[0] == 0:
            return self.cached_training_data
        if self.cached_training_data is None:
            self.cached_training_data = training_data
        else:
            self.cached_training_data = [
                np.vstack((self.cached_training_data[0], training_data[0])),
                np.vstack((self.cached_training_data[1], training_data[1]))]

        data_length = self.cached_training_data[0].shape[0]
        self.logger.debug("size of value net cached training data: %d", data_length)

        if data_length > self.maximum_training_size:
            self.cached_training_data = [
                self.cached_training_data[0][data_length - self.maximum_training_size:],
                self.cached_training_data[1][data_length - self.maximum_training_size:]]

        return self.cached_training_data

    def recall_training_data(self, state_actions, y, predicted_y):
        index = np.where((y - predicted_y) > (y * (1 - self.lbd) / 2))[0]
        self.logger.info("recall records: %s in %s", (index.shape[0], y.shape[0]))
        self.cached_training_data = [state_actions[index], y[index]]

    def train(self, learning_rate, data):
        new_training_data = self.build_td_training_data2(data, learning_rate)
        # if new_training_data[1].shape[0] > 0:
        #     self.save_training_data(new_training_data)
        # train_data = self.merge_with_cached_training_data(new_training_data)
        train_data = new_training_data

        # if train_data is None or train_data[1].shape[0] < self.minimum_training_size:
        #     return 0

        inputs, y = train_data
        return self.train_with_raw_data(inputs, y, learning_rate)

    def validate(self, data):
        validate_data = self.build_td_training_data2(data, 1)
        inputs, y = validate_data
        loss = self.sess.run(self.loss,
                             {self.inputs: inputs,
                              self.y: y,
                              self.mode: learn.ModeKeys.EVAL})
        return loss

    def train_with_raw_data(self, inputs, y, learning_rate, log_epic=10):
        self.logger.info("Value-Net learning rate: %f train_size: %d", learning_rate, inputs.shape[0])

        def eval_epic(epic, loss):
            self.logger.info("epic %d: %f ", epic, loss)
            # idx = np.random.choice(range(y.shape[0]), 20)

            vals = self.sess.run(self.pred, {self.inputs: inputs[0:20],
                                             self.mode: learn.ModeKeys.EVAL})
            result = np.append(vals, np.reshape(y[0:20], vals.shape), axis=1)
            self.logger.info(result)

        # inputs = np.reshape(inputs, [-1, self.board_size[0], self.board_size[1], 1])

        for i in range(self.training_epics):
            pred, loss, _ = self.sess.run([self.pred, self.loss, self.optimizer],
                                          {self.inputs: inputs,
                                           self.y: y,
                                           self.mode: learn.ModeKeys.TRAIN})
            if (i + 1) % log_epic == 0 or i == 0:
                eval_epic(i, loss)
            self.save(self.model_path)

        # self.recall_training_data(state_actions, y, pred)

        return loss

    # def build_td_0_training_data(self, sessions):
    #     inputs = None
    #     y = None
    #     for session in sessions:
    #         steps = len(session)
    #         if steps < 2:
    #             continue
    #
    #         session_inputs = np.array([self.build_state_action(state, action) for state, action, _ in session])
    #         session_y = self.lbd * self.sess.run(self.pred, {self.inputs: session_inputs[1:],
    #                                         self.mode: learn.ModeKeys.EVAL})
    #         for i in range(steps - 1):
    #             state, action, reward = session[i]
    #             if reward == 1:
    #                 session_y[i] = [1]
    #         if inputs is None:
    #             inputs = session_inputs[:-1]
    #         else:
    #             inputs = np.vstack((inputs, session_inputs[:-1]))
    #         if y is None:
    #             y = session_y
    #         else:
    #             y = np.vstack((y, session_y))
    #     return inputs, y

    def build_td_training_data(self, sessions):
        inputs = []
        y = []
        margin = (1 - self.lbd) / 2
        for session in sessions:
            steps = len(session)
            if steps < 2:
                continue
            end_state, end_action, end_reward = session[-1]
            end_value = 1
            if end_action.val != 0:
                continue

            session_inputs = np.array([state for state, _, _ in session])
            session_y = self.sess.run(self.pred, {self.inputs: session_inputs,
                                                  self.mode: learn.ModeKeys.EVAL})

            for index in range(steps - 1, -1, -1):
                state, action, reward = session[index]
                if end_value == 1 and session_y[index] - 1 > margin:
                    inputs.append(state)
                    y.append([end_value])
                elif end_value - session_y[index] > end_value * margin:
                    inputs.append(state)
                    y.append([end_value])
                elif session_y[index] - self.lbd > self.lbd * margin and end_value < 1:
                    inputs.append(state)
                    y.append([end_value])
                end_value = self.lbd * end_value

        return np.array(inputs), np.array(y)

    def build_td_training_data2(self, sessions, learning_rate):
        inputs = []
        y = []
        margin = (1 - self.lbd) / 2
        for session in sessions:
            steps = len(session)
            if steps < 2:
                continue
            end_state, end_action, end_reward = session[-1]
            end_value = 1
            if end_action.val != 0:
                continue

            session_inputs = np.array([state for state, _, _ in session])
            session_y = self.sess.run(self.pred, {self.inputs: session_inputs,
                                                  self.mode: learn.ModeKeys.EVAL})

            for index in range(steps - 1, -1, -1):
                state, action, reward = session[index]
                pred = session_y[index]
                difference = end_value - pred
                # if difference > margin or difference < -margin:
                inputs.append(state)
                y.append([pred + difference * learning_rate])
                end_value = self.lbd * end_value

        return np.array(inputs), np.array(y)

    def save(self, save_path):
        save_file = save_path + "/qvalue_ckpts"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return self.saver.save(self.sess, save_file)

    def restore(self, save_path):
        return self.saver.restore(self.sess, save_path + "/qvalue_ckpts")

    def save_training_data(self, train_data):
        train_file = self.training_data_dir + "/value_net_train.h5"
        if not os.path.exists(self.training_data_dir):
            os.makedirs(self.training_data_dir)
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
            record_count = y.shape[0]
            self.logger.info("Training value net with %d records...", record_count)
            self.train_with_raw_data(inputs,
                                     y,
                                     self.learning_rate, 50)
            # batch_size = 5000
            # batch_count, remain = divmod(record_count, 500)
            # if remain > 0:
            #     batch_count += 1
            # for batch in range(batch_count):
            #     sample = np.random.choice(record_count, batch_size)
            #     self.train_with_raw_data(inputs[sample],
            #                              y[sample],
            #                              self.learning_rate, 50)
