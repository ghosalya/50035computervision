import numpy as np
from libs.layers import *
from libs.fast_layers import *

def conv_output_dim(input_dim, num_filters, filter_size, stride):
    D1, W1, H1 = input_dim
    
    P = (filter_size - 1) / 2
    
    D2 = num_filters
    W2 = (W1 - filter_size + 2 * P)/stride + 1
    H2 = (H1 - filter_size + 2 * P)/stride + 1
    return D2, W2, H2

def pool_output_dim(input_dim, pooling_size):
    pooling_stride = pooling_size
    
    D2, W2, H2 = input_dim
    
    D3 = D2
    W3 = (W2 - pooling_size)/pooling_stride + 1
    H3 = (H2 - pooling_size)/pooling_stride + 1
    return D3, W3, H3

def conv_pool_output_dim(input_dim, num_filters, filter_size, stride, pooling_size):
    D1, W1, H1 = input_dim
    
    P = (filter_size - 1) / 2
    
    D2 = num_filters
    W2 = (W1 - filter_size + 2 * P)/stride + 1
    H2 = (H1 - filter_size + 2 * P)/stride + 1
    
    pooling_stride = pooling_size
    D3 = D2
    W3 = (W2 - pooling_size)/pooling_stride + 1
    H3 = (H2 - pooling_size)/pooling_stride + 1
    return D3, W3, H3

def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU
    
    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Forward pass for the affine-batchnorm-relu convenience layer
    """
    a, fc_cache = affine_forward(x, w, b)
    an, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache

def affine_bn_relu_backward(dout, cache):
    """
    Backward pass for the affine-batchnorm-relu convenience layer
    """
    fc_cache, bn_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dan, dgamma, dbeta = batchnorm_backward(da, bn_cache)
    dx, dw, db = affine_backward(dan, fc_cache)
    return dx, dw, db, dgamma, dbeta

def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.
    
    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db

def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, sbn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, sbn_cache, relu_cache)
    return out, cache

def conv_bn_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, sbn_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dan, dgamma, dbeta = spatial_batchnorm_backward(da, sbn_cache)
    dx, dw, db = conv_backward_fast(dan, conv_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.
    
    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer
    
    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db

def conv_bn_relu_pool_forward(x, w, b, gamma, beta, params):
    conv_param, pool_param, bn_param = params
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, sbn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    s, relu_cache = relu_forward(an)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, sbn_cache, relu_cache, pool_cache)
    return out, cache

def conv_bn_relu_pool_backward(dout, cache):
    conv_cache, sbn_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dan, dgamma, dbeta = spatial_batchnorm_backward(da, sbn_cache)
    dx, dw, db = conv_backward_fast(dan, conv_cache)
    return dx, dw, db, dgamma, dbeta
