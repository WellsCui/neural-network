import numpy as np
import logging
import os
import tensorflow as tf
from tensorflow.contrib import learn, layers
import tables
import game.utils


class AlphaZeroNet(object):
    def __init__(self, name, board_size, learning_rate, lbd):
        self.name = name
        self.board_size = board_size
        self.learning_rate = learning_rate
        # Input Layer
        self.mode = tf.placeholder(tf.string)
        # self.states = tf.placeholder(tf.int8, shape=[batch_size, board_size[0], board_size[1], 1])
        self.inputs = tf.placeholder(tf.float32)
        self.values = tf.placeholder("float")
        self.probabilities = tf.placeholder("float")
        self.lbd = lbd
        self.r = 0.05
        self.kernel_size = [3, 3]
        self.pool_size = [2, 2]
        self.logger = logging.root

        input_layer = tf.reshape(
            self.inputs, [-1, board_size[0], board_size[1], 1],
            name=name + "value_net_input_layer")

        layer1 = self.conv_bn_relu(1, input_layer, 256, self.kernel_size, [1, 1], 'same')
        layer2 = self.conv_bn_relu(2, layer1, 256, self.kernel_size, [1, 1], 'same', False)
        layer3 = self.resAdd_relu(layer1, layer2, 3)
        layer4 = self.conv_bn_relu(4, layer3, 256, self.kernel_size, [1, 1], 'same')
        layer5 = self.conv_bn_relu(5, layer4, 256, self.kernel_size, [1, 1], 'same', False)
        layer6 = self.resAdd_relu(layer4, layer5, 6)
        layer7 = self.conv_bn_relu(7, layer6, 256, self.kernel_size, [1, 1], 'same')
        layer8 = self.conv_bn_relu(8, layer7, 256, self.kernel_size, [1, 1], 'same', False)
        layer9 = self.resAdd_relu(layer7, layer8, 9)

        policy_hd1 = self.conv_bn_relu('policy_hd1', layer9, 2, [1, 1], [1, 1], 'same')
        policy_flat = tf.reshape(policy_hd1, [-1, board_size[0]*board_size[1]*2], name=name + "policy_flat")
        self.policy_output = tf.layers.dense(
            inputs=policy_flat, units=board_size[0]*board_size[1], activation=tf.nn.relu, name=name + "policy_output")

        value_hd1 = self.conv_bn_relu('value_hd1', layer9, 1, [1, 1], [1, 1], 'same')
        value_flat = tf.reshape(value_hd1, [-1, board_size[0]*board_size[1]*1], name=name + "value_flat")
        value_dense1 = tf.layers.dense(
            inputs=value_flat, units=256, activation=tf.nn.relu, name=name + "value_dense1")
        self.value_output = tf.layers.dense(
            inputs=value_dense1, units=1, activation=tf.nn.tanh, name=name + "value_output")

        self.policy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                name=name+"policy_cross_entropy",
                labels=self.probabilities, logits=self.policy_output))

        self.value_loss = tf.losses.mean_squared_error(self.values, self.value_output)

        self.saver = tf.train.Saver()
        # self.losses = layers.l2_regularizer(tf.add(self.policy_loss, self.value_loss, )).
        self.losses = self.policy_loss + self.value_loss

        self.optimizer = tf.train.AdamOptimizer(learning_rate, name=name + "_Optimizer").minimize(self.losses)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

    def conv_bn_relu(self, layer, input, filters, kernel_size, stride, padding, use_relu=True):
        conv = tf.layers.conv2d(
            name=self.name + "value_net_conv"+str(layer),
            inputs=input,
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
        )
        bn1 = tf.layers.batch_normalization(conv, training=self.mode == learn.ModeKeys.TRAIN, name=self.name + "value_net_bn"+str(layer))
        if use_relu:
            return tf.nn.relu(bn1, name="value_net_relu"+str(layer))
        else:
            return bn1

    def resAdd_relu(self, a, b, layer):
        # a_shape = tf.shape(a)
        # b_shape = tf.shape(b)
        return tf.nn.relu(tf.add(a, b, self.name+"_value_net_rest_add_"+str(layer)), self.name+"_value_net_rest_relu_"+str(layer))

    def evaluate(self, state):
        new_state = state.copy()
        return self.batch_evaluate(new_state)

    def batch_evaluate(self, states):
        result = self.sess.run([self.policy_output, self.value_output], {self.inputs: states,
                                         self.mode: learn.ModeKeys.EVAL})
        return [(result[0][i], result[1][i]) for i in range(len(states))]

    def save(self, save_path):
        save_file = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return self.saver.save(self.sess, save_file)

    def restore(self, save_path):
        return self.saver.restore(self.sess, save_path)

    def train(self, train_data, epics=100):
        inputs, probabilities, values = train_data
        policy_loss, value_loss = 0, 0
        for i in range(epics):
            policy_loss, value_loss, losses, _ = self.sess.run([self.policy_loss, self.value_loss, self.losses, self.optimizer],
                                          {self.inputs: inputs,
                                           self.probabilities: probabilities,
                                           self.values: values,
                                           self.mode: learn.ModeKeys.TRAIN})

        return policy_loss, value_loss
