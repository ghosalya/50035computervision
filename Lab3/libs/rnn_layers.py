import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.
    
    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.
    
    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)
    
    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    # next_h = fw(prev_h, x) = tanh(Whh*prev_h + Whx*x + b)
    next_h = np.tanh(np.matmul(prev_h, Wh) + np.matmul(x, Wx) + b) 
    # cache
    cache = (x, prev_h, Wx, Wh, b, next_h)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.
    
    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass
    
    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    x, prev_h, Wx, Wh, b, next_h = cache
    dtanh = (1 - next_h**2)
    dx = np.matmul(dnext_h*dtanh, Wx.T)
    dprev_h = np.matmul(dnext_h*dtanh, Wh.T)
    dWh = np.matmul(prev_h.T, dnext_h*dtanh)
    dWx = np.matmul(x.T, dnext_h*dtanh)
    db = np.sum(dnext_h*dtanh, axis=0)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.
    
    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)
    
    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above.                                                                     #
    ##############################################################################
    N, T, D = x.shape
    N, H = h0.shape
    h = np.zeros((N, T+1, H)) # T+1 for storing H0
    h[:,0,:] = h0
    cache = []
    for i in range(x.shape[1]):
        x_ = x[:,i,:] # one timeframe
        next_h, ch = rnn_step_forward(x_, h[:,i,:], Wx, Wh, b)
        cache.append(ch)
        h[:,i+1,:] = next_h

    h = h[:,1:,:] # remove h0
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.
    
    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)
    
    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    # dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above.                                                             #
    ##############################################################################
    N, D = cache[0][0].shape
    N, T, H = dh.shape

    current_dh = dh

    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx = np.zeros((D, H))
    dWh = np.zeros((H, H))
    db = np.zeros(H)

    # dual loop backprop, why tho?
    for t in range(T):
        dx_t, dh_t, dWx_t, dWh_t, db_t = rnn_step_backward(current_dh[:,t,:], cache[t])
        current_dht = dh_t
        if t == 0:
            dh0 += dh_t
        for i in range(t-1, -1, -1):
            dx_i, dh_i, dWx_i, dWh_i, db_i = rnn_step_backward(current_dht, cache[i])
            current_dht = dh_i
            if i == 0:
                dh0 += dh_i
            dx[:,i,:] += dx_i
            dWh_t += dWh_i
            dWx_t += dWx_i
            db_t += db_i
        dx[:,t,:] += dx_t
        dWx += dWx_t
        dWh += dWh_t
        db += db_t

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.
    
    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.
    
    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This should be very simple.                                          #
    ##############################################################################
    N, T = x.shape
    V, D = W.shape
    cache = (x, W)
    out = np.zeros((N, T, D))
    for i in range(N):
        for j in range(T):
            idx = x[i,j]
            out[i,j,:] += W[idx,:]
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.
    
    HINT: Look up the function np.add.at
    
    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass
    
    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    x, W = cache
    N, T, D = dout.shape
    V, D = W.shape

    dW = np.zeros((V, D))
    for i in range(N):
        for j in range(T):
            idx = x[i,j]
            dW[idx,:] += dout[i,j,:]
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.
    
    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.
    
    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Any matrix of 4H is of the following:
    it = [0-H]
    ft = [H-2H]
    ot = [2H-3H]
    ~Ct = [3H-4H]
    
    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    N, D = x.shape
    N, H = prev_h.shape

    # 1. forget
    Wfx, Wfh, bf = Wx[:,H:2*H], Wh[:,H:2*H], b[H:2*H]
    fxt = sigmoid(np.matmul(x, Wfx) + np.matmul(prev_h, Wfh) + bf)

    # 2. update
    Wix, Wih, bi = Wx[:,0:H], Wh[:,0:H], b[0:H]
    ixt = sigmoid(np.matmul(x, Wix) + np.matmul(prev_h, Wih) + bi)
    Wcx, Wch, bc = Wx[:,3*H:4*H], Wh[:,3*H:4*H], b[3*H:4*H]
    cxt = np.tanh(np.matmul(x, Wcx) + np.matmul(prev_h, Wch) + bc)

    # cell update
    next_c = np.multiply(fxt, prev_c) + np.multiply(ixt, cxt)

    # 3. output
    Wox, Woh, bo = Wx[:,2*H:3*H], Wh[:,2*H:3*H], b[2*H:3*H]
    oxt = sigmoid(np.matmul(x, Wox) + np.matmul(prev_h, Woh) + bo)

    # state update
    next_h = np.multiply(oxt, np.tanh(next_c))

    cache = (N, H, D, x, prev_h, prev_c, Wx, Wh, b, fxt, ixt, cxt, oxt) # modify later
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    
    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.
    
    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass
    
    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    # x, prev_h, Wx, Wh, b, next_h = cache
    # dtanh = (1 - next_h**2)
    # dx = np.matmul(dnext_h*dtanh, Wx.T)
    # dprev_h = np.matmul(dnext_h*dtanh, Wh.T)
    # dWh = np.matmul(prev_h.T, dnext_h*dtanh)
    # dWx = np.matmul(x.T, dnext_h*dtanh)
    # db = np.sum(dnext_h*dtanh, axis=0)

    N, H, D, x, prev_h, prev_c, Wx, Wh, b, fxt, ixt, cxt, oxt = cache
    # print(N, H, D, Wx.shape)
    next_c = np.multiply(fxt, prev_c) + np.multiply(ixt, cxt)
    _next_c = 1 - np.tanh(next_c)**2

    dfxt = fxt*(1-fxt) * prev_c * (dnext_c + oxt * _next_c * dnext_h)
    dixt = ixt*(1-ixt) * cxt * (dnext_c + oxt * _next_c * dnext_h)
    dcxt = (1-cxt**2) * ixt * (dnext_c + oxt * _next_c * dnext_h)
    doxt = oxt*(1-oxt) * dnext_h * np.tanh(next_c)

    # only cxt is tanh
    # dtanh = 1-tanh^2 ; dsig = sig*(1-sig)

    dgates = np.hstack((dixt, dfxt, doxt, dcxt))

    dx = np.matmul(dgates, Wx.T)
    dprev_h = np.matmul(dgates, Wh.T)
    dprev_c = dnext_c * fxt + oxt * _next_c * fxt * dnext_h
    dWx = np.matmul(x.T, dgates)
    dWh = np.matmul(prev_h.T, dgates)
    db = np.sum(dgates, axis=0)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    
    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b, c0=None):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.
    
    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.
    
    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)
    
    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    N, T, D = x.shape
    N, H = h0.shape
    h = np.zeros((N, T+1, H)) # T+1 for storing H0
    c = np.zeros((N, T+1, H)) # T+1 for storing H0
    h[:,0,:] = h0
    if c0 is not None:
        c[:,0,:] = c0
    cache = []
    for i in range(x.shape[1]):
        x_ = x[:,i,:] # one timeframe
        next_h, next_c, ch = lstm_step_forward(x_, h[:,i,:], c[:,i,:], Wx, Wh, b)
        cache.append(ch)
        h[:,i+1,:] = next_h
        c[:,i+1,:] = next_c
    c = c[:,1:,:]
    cache.append(c)

    h = h[:,1:,:] # remove h0
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.
    
    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass
    
    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    N, H, D = cache[0][:3]
    N, T, H = dh.shape

    # print(N, H, T, D)

    current_dh = dh
    current_c = cache[-1]

    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx = np.zeros((D, 4*H))
    dWh = np.zeros((H, 4*H))
    db = np.zeros(4*H)

    current_dht = 0
    current_dc = 0
    for i in range(T-1, -1, -1):
        dxi, dprev_hi, dprev_ci, dWxi, dWhi, dbi = lstm_step_backward(current_dh[:,i,:] + current_dht, 
                                                                      current_dc, cache[i])
        current_dht = dprev_hi
        current_dc = dprev_ci
        if i == 0:
            dh0 += dprev_hi
        dx[:,i,:] += dxi
        dWh += dWhi
        dWx += dWxi
        db += dbi
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    
    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.
    
    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)
    
    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.
    
    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass
    
    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]
    
    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))
    
    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.
    
    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.
    
    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.
    
    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """
    
    N, T, V = x.shape
    
    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)
    
    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]
    
    if verbose: print('dx_flat: ', dx_flat.shape)
    
    dx = dx_flat.reshape(N, T, V)
    
    return loss, dx

