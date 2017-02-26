import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ResNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - bn - relu - 2x2 max pool- {-- conv - bn - relu - conv - bn - relu -}*N - dropout - affine - softmax
                                      |______________________________________|
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32),
                 num_classes=10, weight_scale=1e-3, reg=0.001, dropout=0.7,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.params_ = {}
        self.weight_scale = weight_scale

        ############################################################################
        # TODO: Initialize weights and biases for the Residual convolutional       #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################

        conv_metadatas = np.array([[16, 3, 1],
                                   # [64, 3, 1],
                                   # [64, 3, 1],
                                   # [64, 3, 1],
                                   [16, 3, 1],
                                   [32, 3, 1],
                                   # [128, 3, 1],
                                   # [128, 3, 1],
                                   [32, 3, 1],
                                   # [64, 3, 1],
                                   # [256, 3, 1],
                                   # [256, 3, 1],
                                   # [64, 3, 1],
                                   # [128, 3, 1],
                                   # [512, 3, 1],
                                   # [512, 3, 1],
                                   # [128, 3, 1]
                                   ])

        conv_input_dim = input_dim
        self.conv_layers = conv_metadatas.shape[0]
        for i in range(self.conv_layers):
            num_filters, filter_size, stride = conv_metadatas[i]
            conv_layer_params, self.params_[i], conv_input_dim = self.create_conv_bn_params(conv_input_dim, num_filters,
                                                                                            filter_size, stride,
                                                                                            self.weight_scale)
            i_str = str(i)
            w, b, gamma, beta = conv_layer_params
            self.params['w' + i_str] = w
            self.params['b' + i_str] = b
            self.params['gamma' + i_str] = gamma
            self.params['beta' + i_str] = beta

        self.params['w'+str(self.conv_layers)] = np.random.randn(np.prod(conv_input_dim), num_classes) * weight_scale
        self.params['b'+str(self.conv_layers)] = np.zeros([num_classes])

        self.dropout_param = {'mode': 'train', 'p': dropout}

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k in self.params:
            self.params[k] = self.params[k].astype(dtype)

    def create_conv_bn_params(self, input_dim, num_filters, filter_size, stride, weight_scale):
        w = np.random.randn(num_filters, input_dim[0], filter_size, filter_size) * weight_scale
        b = np.zeros([num_filters])
        pad = int((filter_size - 1) / 2)
        conv_param = {'stride': stride, 'pad': pad}
        HH = 1 + (input_dim[1] + 2 * pad - filter_size) / stride
        WW = 1 + (input_dim[2] + 2 * pad - filter_size) / stride

        gamma = np.ones([num_filters * HH * WW])
        beta = np.zeros([num_filters * HH * WW])
        bn_params = {'mode': 'train'}

        return (w, b, gamma, beta), (conv_param, bn_params), (num_filters, HH, WW)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the Residual convolutional network.

        """
        scores, cache = self.forward(X, y)
        if y is None:
            return scores

        loss, dout = softmax_loss(scores, y)
        return self.backward(loss, dout, cache)

    def forward(self, X, y):
        res = np.zeros(X.shape)
        input = X
        cache = {}
        for i in range(self.conv_layers):
            i_str = str(i)
            if np.remainder(i, 2) == 0:
                input = self.add_residual(input, res)
                res = input

            w = self.params['w' + i_str]
            b = self.params['b' + i_str]
            gamma = self.params['gamma' + i_str]
            beta = self.params['beta' + i_str]

            conv_param, bn_params = self.params_[i]
            if y is None:
                bn_params['mode'] = 'test'
            else:
                bn_params['mode'] = 'train'
            input, cache[i] = conv_bn_relu_forward(input, w, b, conv_param, gamma, beta, bn_params)

        if y is None:
            self.dropout_param['mode'] = 'test'
        else:
            self.dropout_param['mode'] = 'train'

        input, cache['dropout'] = dropout_forward(input, self.dropout_param)
        w = self.params['w' + str(self.conv_layers)]
        b = self.params['b' + str(self.conv_layers)]
        scores, cache[self.conv_layers] = affine_forward(input, w, b)
        return scores, cache

    def backward(self, loss, dout, cache):
        grads = {}
        dout, dw, db = affine_backward(dout, cache[self.conv_layers])
        grads['w' + str(self.conv_layers)] = dw + self.reg * self.params['w' + str(self.conv_layers)]
        grads['b' + str(self.conv_layers)] = db
        dout = dropout_backward(dout, cache['dropout'])
        loss += 0.5 * self.reg * np.sum(self.params['w' + str(self.conv_layers)]**2)
        dres = np.zeros(dout.shape)

        for i in range(self.conv_layers)[::-1]:
            i_str = str(i)
            if np.remainder(i, 2) == 0:
                dout = self.add_residual(dout, dres)
                dres = dout

            dout, dw, db, dgamma, dbeta = conv_bn_relu_backward(dout, cache[i])
            loss += 0.5 * self.reg * np.sum(self.params['w' + i_str]**2)
            grads['w' + i_str] = dw + self.reg * self.params['w' + i_str]
            grads['b' + i_str] = db
            grads['gamma' + i_str] = dgamma
            grads['beta' + i_str] = dbeta

        return loss, grads

    def add_residual(self, x, res):
        x_shape = x.shape
        res_shape = res.shape
        if x_shape == res_shape:
            x += res
        else:
            x_dim = np.prod(x_shape[1:])
            res_dim = np.prod(res_shape[1:])
            w_ = np.zeros((res_dim, x_dim))
            if x_dim>res_dim:
                w_[0:res_dim, 0:res_dim] = np.identity(res_dim)
            else:
                w_[0:x_dim, 0:x_dim] = np.identity(x_dim)
            projected_res = res.reshape((x_shape[0], res_dim)).dot(w_)
            x += projected_res.reshape(x_shape)
        return x

    # def add_dresidual(self, dx, dres):
    #     dx_shape = dx.shape
    #     dres_shape = dres.shape
    #     if dx_shape == dres_shape:
    #         dx += dres
    #     else:
    #         x_dim = np.prod(dx_shape[1:])
    #         out_dim = np.prod(dres_shape[1:])
    #         w_ = np.zeros((x_dim, out_dim))
    #         w_[0:x_dim, 0:x_dim] = np.identity(x_dim)
    #         projected_x = x.reshape((dx_shape[0], x_dim)).dot(w_)
    #         dx += projected_x.reshape(dx_shape)
    #     return dx