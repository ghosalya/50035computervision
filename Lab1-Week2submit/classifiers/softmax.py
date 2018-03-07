import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
    
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
    
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    
    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    
    # softmax function
    loss_matrix = X.dot(W)
    for i in range(len(loss_matrix)):
        maxx = loss_matrix[i].max()
        exped = [np.exp(e - maxx) for e in loss_matrix[i]]
        loss_matrix[i] = [e/sum(exped) for e in exped]
        # add only the test value to loss
        loss -= np.log(loss_matrix[i][y[i]])

    # compute gradient
    for i in range(X.shape[0]):
        # looping through inputs
        dLi_dW = np.zeros(W.shape)
        for m in range(W.shape[1]):
            # looping through classes
            Pyi = loss_matrix[i,y[i]]
            Pm = loss_matrix[i,m]
            if m == y[i]:
                dLi_dW[:,m] = ((Pyi-1) * X[i].T)
            else:
                dLi_dW[:,m] = (Pm * X[i].T)
        dW += dLi_dW

    dW = dW / X.shape[0]

    dW = dW + reg*W

    loss = loss/len(X) + 0.5*reg*(np.linalg.norm(W)**2)

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    loss_matrix = X.dot(W)
    loss_matrix = np.exp(loss_matrix - np.max(loss_matrix, axis=1, keepdims=True))
    loss_matrix = loss_matrix / np.sum(loss_matrix, axis=1, keepdims=True)

    loss = -np.sum(np.log(loss_matrix[np.arange(X.shape[0]), y]))/X.shape[0] + 0.5*reg*(np.linalg.norm(W)**2)

    y_oh = np.zeros((y.shape[0], W.shape[1]))
    y_oh[np.arange(y.shape[0]), y] = 1
    sub_loss_matrix = loss_matrix - y_oh
    dW = X.T.dot(sub_loss_matrix)/X.shape[0] + reg*W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    
    return loss, dW

