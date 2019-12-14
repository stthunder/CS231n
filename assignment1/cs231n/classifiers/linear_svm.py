from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]

    loss = 0.0

    x = np.zeros((num_train,num_classes))
    xx = np.zeros((num_train, num_classes))

    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i].T
                dW[:, j] += X[i].T
                # x[i,j]  = margin
                # xx[i,j] = y[i]
                # dW[b, c] = dW[b, c] + X[j, b]
                #
                # dW[b, int(xx[j, c])] = dW[b, int(xx[j, c])] - X[j, int(xx[j, c])]
        """
                for c in range(W.shape[0]):
                    dW[c,j] = dW[c,j] + X[i,c]
                    dW[c,int(y[i])] = dW[c,int(y[i])] - X[i,c]
            # else:
            #     x[i,j] = 0
        """



    dW /= num_train
    dW  = dW + 2 * W * reg
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    score = X.dot(W)
    index = np.arange(X.shape[0])
    score_correct  = score[index,y]

    score_caculate = np.tile(score_correct,(W.shape[1],1))#用了太多广播机制
    score_caculate = score_caculate.T#转换起来过于麻烦
    score = score -score_caculate + np.ones(score.shape)
    lesszeros = np.argwhere(score<0)
    morezeros = np.argwhere(score>0)
    equalzeros = np.argwhere(score==1)
    #用np.maximum直接返回比较后结果
    loss = (sum(score[morezeros[:,0],morezeros[:,1]])-equalzeros.shape[0])/X.shape[0] + reg * np.sum(W*W)


    score_deta = np.zeros(score.shape)
    score_deta[morezeros[:,0],morezeros[:,1]] = 1

    # for i in range (W.shape[1]):
        # dW[:,i] = (np.sum((score_deta[:,i]).T*X,axis = 0)).T


    # minus = (X.T)*score_deta
    # dW_t    = minus
    # print(dW-dW_t)


    re_X = X.reshape((X.shape[0],X.shape[1],1))
    re_X = np.tile(re_X,(1,W.shape[1]))

    score_X = np.tile(score_deta,(1,X.shape[1]))
    score_X = np.reshape(score_X,re_X.shape)

    dW   = re_X*score_X
    dW   = np.sum(dW,axis=0)
    #总的来说计算时间还是很长
    ##################################################


    zz   = np.zeros((X.shape[0],score.shape[1]))
    zz[range(y.shape[0]),y] = 1
    zz = np.tile(zz,[1,X.shape[1]])

    ysum = np.sum(score_deta,axis=1)
    ysum = np.tile(ysum,(X.shape[1],1))


    re_X = X.reshape((X.shape[0], X.shape[1], 1))


    ysum = np.reshape(ysum.T,re_X.shape)
    ysum = re_X*ysum
    ysum = np.tile(ysum,[1,score_deta.shape[1]])
    zz   = np.reshape(zz,ysum.shape)
    zz   = zz*ysum
    zz   = np.sum(zz,axis=0)

    dW   = dW-zz








    # print(score[[1,1]],score[1,1],score[morezeros[1]])
    # loss = sum
    # print((score[int(morezeros[:,0]),int(morezeros[:,1])]))
    # print(loss)
    # dW[]
    #加的
    # m = (np.sum(X,axis = 0))
    # print(m.shape)
    # m = m.reshape((m.shape[0],1))
    # print(m.shape)
    #
    # M = np.tile(m,(1,W.shape[1]))
    # print(M.shape)
    # dW = dW + M
    # #减的
    # K = np.zeros(W.shape)
    # m = (np.sum(X, axis=0))
    #找个数 这里可以用lambda找每个出现的个数
    # for i  in range(W.shape[1]):
    #     num = np.argwhere(y == i).shape[0]
    #     K[:,i] = m*num
    #     print(num)
    # print(K)
    # dW = dW - K

    #最后再归零即可
    # dW[lesszeros[:,0],lesszeros[:,1]] = 0

    dW /= X.shape[0]
    # print(dW[:,morezeros[:,1]].shape,dW.shape,morezeros[:,1].shape)
    #= dW[:,morezeros[:,1]] + X[morezeros[:,0],:].T

    # dW[:,y[morezeros[:, 0]]] = dW[:,y[morezeros[:, 0]]] - X[morezeros[:, 0], :]


    dW = dW + 2 * W * reg

    # print(score_caculate.shape)


    # margin = score

    """


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
