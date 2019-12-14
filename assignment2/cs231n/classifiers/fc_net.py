from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # self.params['W1'] =  np.random.normal(loc=0.0, scale=1/(input_dim*hidden_dim), size=(input_dim,hidden_dim))/np.sqrt(input_dim*hidden_dim)*weight_scale
        self.params['W1'] =  np.random.randn(input_dim,hidden_dim)*weight_scale
        self.params['W2'] =  np.random.randn(hidden_dim,num_classes)*weight_scale
        self.params['b1'] =  np.zeros([hidden_dim,])
        self.params['b2'] =  np.zeros([num_classes,])
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        X_rs = np.reshape(X,(X.shape[0],-1))
        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']

        Xrs = np.reshape(X,(X.shape[0],-1))
        first = Xrs.dot(W1)+b1
        relu_ans = np.maximum(first,0)
        scores = relu_ans.dot(W2) + b2



        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores


        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, grads_soft = softmax_loss(scores,y)
        loss += self.reg*(np.sum(W1*W1)+np.sum(W2*W2))*0.5

        cache = relu_ans, W2, b2
        dx,dw,db = affine_backward(grads_soft, cache)

        grads['W2'] = dw + self.reg * W2
        grads['b2'] = db
        gradsecond  = dx

        gradsrelu = (relu_backward(gradsecond,first))


        cache = X,W1,b1
        dx,dw,db = affine_backward(gradsrelu,cache)
        grads['W1'] = dw + self.reg * W1
        grads['b1'] = db

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


        for i in range(self.num_layers):
            W1 = ('W%d') % (i+1)
            b1 = ('b%d') % (i+1)
            if(i == 0):
                self.params[W1] = np.random.randn(input_dim,hidden_dims[0])*weight_scale
                self.params[b1] = np.zeros([hidden_dims[0],])
            elif(i == self.num_layers-1):
                self.params[W1] = np.random.randn(hidden_dims[i-1],num_classes)*weight_scale
                self.params[b1] = np.zeros([num_classes,])
            else:
                self.params[W1] = np.random.randn(hidden_dims[i-1],hidden_dims[i])*weight_scale
                self.params[b1] = np.zeros([hidden_dims[i],])
            if(i != self.num_layers - 1):
                if (self.normalization == "batchnorm"):
                    Scale = ('Scale%d')% (i+1)
                    shift = ('shift%d') % (i+1)
                    self.params[Scale] = np.ones_like(self.params[b1])
                    self.params[shift] = np.zeros_like(self.params[b1])
                if (self.normalization == "layernorm"):
                    Scale = ('Scale%d') % (i + 1)
                    shift = ('shift%d') % (i + 1)
                    self.params[Scale] = np.ones_like(self.params[b1])
                    self.params[shift] = np.zeros_like(self.params[b1])
                # if(dropout):

                # if (self.normalization == "batchnorm"):





        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        #初始normalization
        # X_mean = np.mean(X, axis = 0)
        # X = np.std (X, axis = 0)

        # np.random.seed(self.dropout_param['seed'])
        #!!!
        # for i in range(self.num_layers):

        #生成随机阵

        after_relu = X
        save  = {}
        cache = {}
        sum_reg = 0
        #未处理SOFTMAX
        for i in range(self.num_layers-1):
            b1 = ('b%d') % (i + 1)
            W1 = ('W%d') % (i + 1)

            sum_reg = sum_reg + self.reg*np.sum(self.params[W1]*self.params[W1])*0.5

            first_affine,a = affine_forward(after_relu,self.params[W1],self.params[b1])
            #affine
            cache[b1] = a
            if(self.normalization=="batchnorm"):
                Scale = ('Scale%d')% (i+1)
                shift = ('shift%d')% (i+1)
                first_affine, Cache = batchnorm_forward(first_affine,self.params[Scale],self.params[shift], self.bn_params[i])
                self.params[Scale] = Cache[1]
                self.params[shift] = Cache[2]
                cache[Scale] = Cache

            if (self.normalization == "layernorm"):
                Scale = ('Scale%d')% (i+1)
                shift = ('shift%d')% (i+1)
                first_affine, Cache = layernorm_forward(first_affine,self.params[Scale],self.params[shift], self.bn_params[i])
                self.params[Scale] = Cache[1]
                self.params[shift] = Cache[2]
                cache[Scale] = Cache



                # print(Cache[6])
                # bn_param = Cache[6]
                # Scale = ('Scale % d')% (i+1)
                # shift  = ('shift % d') % (i+1)
                #
                # save[W1] = np.mean(first_affine,axis=0)
                # save[b1] = np.std (first_affine,axis=0)
                #
                # first_affine -= save[W1]
                # first_affine /= save[b1]+1e-5
                #
                # first_affine = first_affine*self.params[Scale] + self.params[shift]
            #BN层


            after_relu,a = relu_forward(first_affine)
            cache[W1] = a

            if self.use_dropout:
                Drop = ('Drop%d') % (i+1)
                after_relu, Cache = dropout_forward(after_relu, self.dropout_param)
                cache[Drop] = Cache


            #Relu层

            # if self.use_dropout:
            #     rand = np.random.rand(self.params[b1].shape[0])<0.5
            #     rand += 0
            #
            # # 生成随机阵
            #     after_relu *=rand

            #Dropout层

        b1 = ('b%d') % (self.num_layers)
        W1 = ('W%d') % (self.num_layers)

        scores,a = affine_forward(after_relu, self.params[W1], self.params[b1])
        sum_reg = sum_reg + self.reg * np.sum(self.params[W1] * self.params[W1]) * 0.5


        #Softmax





        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss,dx = softmax_loss(scores,y)
        loss += sum_reg

        #反向传播

        dx, dw, db = affine_backward(dx, a)

        grads[W1] = dw + self.reg*self.params[W1]
        grads[b1] = db


        for i in range(self.num_layers-1):

            index = self.num_layers -i -1  #反向传播


            if self.use_dropout:
                Drop = ('Drop%d')%(index)
                dx = dropout_backward(dx, cache[Drop])


            b1 = ('b%d') % (index)
            W1 = ('W%d') % (index)
            dx = relu_backward(dx,cache[W1])


            if(self.normalization=="batchnorm"):
                Scale = ('Scale%d') % (index)
                shift = ('shift%d') % (index)

                dx, grads[Scale], grads[shift] = batchnorm_backward(dx, cache[Scale])

            if(self.normalization=="layernorm"):
                Scale = ('Scale%d') % (index)
                shift = ('shift%d') % (index)

                dx, grads[Scale], grads[shift] = layernorm_backward(dx, cache[Scale])


                # first_affine,Cache = dropout_forward(first_affine, self.dropout_param)


            dx, dw, db = affine_backward(dx, cache[b1])

            grads[W1] = dw + self.reg*self.params[W1]
            grads[b1] = db





        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
