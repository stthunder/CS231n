from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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

    ans = X.dot(W)

    for i in range(X.shape[0]):
        f  = ans[i]
        f -= np.max(f)
        fenmu = np.sum(np.exp(f))
        p    = np.exp(f[y[i]]) / fenmu
        loss = loss - np.log(p)
        dW[:,y[i]] = dW[:,y[i]] - X[i]
        for j in range(W.shape[1]):
            dW[:, j] = dW[:, j] + X[i]*(np.exp(ans[i, j])/fenmu)


    num_train = X.shape[0]
    dW /= num_train
    dW  = dW + 2 * W * reg
    loss /= num_train
    loss += reg * np.sum(W * W)
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ans = X.dot(W)


    maxofans = np.max(ans,axis=1)
    maxofans = maxofans.reshape((ans.shape[0],1))
    maxofans = np.tile(maxofans,(1,ans.shape[1]))
    ans = ans - maxofans
    ans = np.exp(ans)
    sumexp  = np.sum(ans,axis=1)
    index = np.arange(X.shape[0])
    realnum = ans[index,y]
    loss    = -np.sum(np.log(realnum/sumexp))


    frame  = np.reshape(X,(((X.shape[0])*(X.shape[1])),1))
    frame  = np.tile(frame,(1,W.shape[1]))
    frame  = np.reshape(frame,(X.shape[0],X.shape[1],W.shape[1]))
    indexreal = np.zeros_like(ans)
    indexreal[index,y] =  1
    indexreal = np.tile(indexreal,(1,X.shape[1]))
    indexreal = np.reshape(indexreal,frame.shape)
    indexreal = indexreal * frame
    indexreal = np.reshape(indexreal,(X.shape[0],X.shape[1],W.shape[1]))
    indexreal = np.sum(indexreal, axis=0)
    dW = dW - indexreal

    sumexp   = np.reshape(sumexp,(sumexp.shape[0],1))
    sumexp   = np.tile(sumexp,(1,W.shape[1]))
    everyexp = ans/sumexp
    everyexp = np.tile(everyexp,(1,X.shape[1]))
    everyexp = np.reshape(everyexp,frame.shape)
    everyexp = everyexp * frame
    indexreal = np.reshape(everyexp,(X.shape[0],X.shape[1],W.shape[1]))
    indexreal = np.sum(indexreal, axis=0)
    dW = dW + indexreal






    num_train = X.shape[0]
    dW /= num_train
    dW  = dW + 2 * W * reg
    loss /= num_train
    loss += reg * np.sum(W * W)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
