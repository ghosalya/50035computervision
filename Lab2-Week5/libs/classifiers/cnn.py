import numpy as np

from libs.layers import *
from libs.fast_layers import *
from libs.layer_utils import *

class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    
    conv - relu - 2x2 max pool - affine - relu - affine - softmax
    
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
  
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
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
    
        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        # conv
        self.params['W1'] = weight_scale * np.random.randn(num_filters, input_dim[0],
                                                           filter_size, filter_size)
        self.params['b1'] = weight_scale * np.zeros(num_filters)

        # first affine - assume stride 1 and appropriate padding
        conv_output_dim = int(np.prod(input_dim[1:])/4 * num_filters)
        self.params['W2'] = weight_scale * np.random.randn(conv_output_dim, hidden_dim)
        self.params['b2'] = weight_scale * np.zeros(hidden_dim)

        # last affine
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = weight_scale * np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
     
 
    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        
        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        
        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    
        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        x_conv, cache_conv = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        x_a1, cache_a1 = affine_relu_forward(x_conv, W2, b2)
        x_a2, cache_a2 = affine_forward(x_a1, W3, b3)

        scores = x_a2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        
        if y is None:
            return scores
        
        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, sfdx = softmax_loss(scores, y)
        loss += self.reg * 0.5 * (np.linalg.norm(self.params['W2'])**2 + np.linalg.norm(self.params['W3'])**2)
        loss += self.reg * 0.5 * np.linalg.norm(self.params['W1'])

        dx_a2, dw_a2, db_a2 = affine_backward(sfdx, cache_a2)
        dw_a2 += self.reg * self.params['W3'] # add l2reg gradient

        dx_a1, dw_a1, db_a1 = affine_relu_backward(dx_a2, cache_a1)
        dw_a1 += self.reg * self.params['W2'] # add l2reg gradient

        dx_conv, dw_conv, db_conv = conv_relu_pool_backward(dx_a1, cache_conv)
        dw_conv += self.reg * self.params['W1']

        grads = {'W1': dw_conv, 'b1': db_conv,
                 'W2': dw_a1, 'b2': db_a1,
                 'W3': dw_a2, 'b3': db_a2,}
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        
        return loss, grads

