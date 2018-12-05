from dnn_utils_v2 import *
import numpy as np
import h5py
import matplotlib.pyplot as plt


def initialize_parameters_deep(X_size, layers):
    """
    Arguments:
    X_size -- входной размер для нейронки
    layers -- [(activation, размер_слоя), ...]
    
    activation: 'relu', 'sigmoid', 'relu_pow'
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    np.random.seed(1)
    parameters = {}
    prev_size = X_size

    print(X_size, layers)
    idx = 0
    for (activation, layer_dims) in layers:
        idx += 1
        parameters['W' + str(idx)] = np.random.randn(layer_dims, prev_size) / np.sqrt(prev_size)
        parameters['b' + str(idx)] = np.zeros((layer_dims, 1))
        prev_size = layer_dims
#         assert(parameters['W_' + str(idx)].shape == (layer_dims, prev_size))
#         assert(parameters['b_' + str(idx)].shape == (layer_dims, 1))

    idx = 0
    for (activation, layer_dims) in layers:
        idx += 1
        if activation == "relu_pow":
            parameters['k' + str(idx)] = np.full((layer_dims, 1), 1.0)
        if activation == "relu_leak":
            parameters['k' + str(idx)] = np.full((layer_dims, 1), 0.2)
    
    return parameters


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    Z = W.dot(A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, k, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    elif activation == "relu_pow":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu_pow(Z, k)
    
    elif activation == "relu_leak":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu_leak(Z, k)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward2(X, parameters, layers):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(layers)
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(0, L):
        A_prev = A
        k = None
        W = parameters["W" + str(l+1)]
        b = parameters["b" + str(l+1)]
        if ("k" + str(l+1)) in parameters: 
            k = parameters["k" + str(l+1)]
        
#         print(A.shape, "x", W.shape, " => ", layers[l])
        A, cache = linear_activation_forward(A_prev, W, b, k, layers[l][0])
        caches.append(cache)
    
#     # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
#     ### START CODE HERE ### (≈ 2 lines of code)
#     AL, cache = linear_activation_forward(A, 
#                                           parameters["W_" + str(L)], 
#                                           parameters["b_" + str(L)],
#                                           parameters["k_" + str(L)], 
#                                           layers[L][0])
#     caches.append(cache)
#     ### END CODE HERE ###
    
#     проверка количества примеров
    out_dims = layers[-1][1] # количество выходов
    assert(A.shape == (out_dims, X.shape[1]))
            
    return A, caches

def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost

def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    dZ = None
    dk = None
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        
    elif activation == "relu_pow":
        dZ, dk = relu_pow_backward(dA, activation_cache)
        
    elif activation == "relu_leak":
        dZ, dk = relu_leak_backward(dA, activation_cache)
    
    assert(dZ is not None)
    
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db, dk

def L_model_backward(AL, Y, caches, layers):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(layers) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    ### START CODE HERE ### (1 line of code)
#     dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    dAL = np.divide(1 - Y, 1 - AL) - np.divide(Y, AL)
    ### END CODE HERE ###
    
    last_dA = dAL
    for l in reversed(range(L)):
        if np.isnan(last_dA).any():
#             print("WTF", last_dA)
            for i in range(last_dA.shape[1]):
                if np.isnan(last_dA[:, i]).any():
#                     print("NaN is fixed", i)
                    last_dA[:, i] = 0
#             assert(False)
            
        activation = layers[l][0]
        prev_dA, dW, db, dk = linear_activation_backward(last_dA, caches[l], activation)

        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db
        if dk is not None:
            grads["dk" + str(l + 1)] = dk
        last_dA = prev_dA

    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W_" + str(l)] = ... 
                  parameters["b_" + str(l)] = ...
    """
    
    for key in parameters:
        dKey = "d"+key
        if dKey in grads:
#             print(grads[dKey], parameters[key])
#             print(type(grads[dKey]), type(parameters[key]))
#             if dKey == "dk1":
#                 print(grads[dKey], parameters[key])
            parameters[key] -= learning_rate * grads[dKey]
            
    return parameters

def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False, onIteration = None):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
    ### START CODE HERE ###
    layers = list(map(lambda x: ("relu", x), layers_dims))
    layers[-1] = ("sigmoid", layers[-1][1])
    layers.pop(0)
    print(layers)
    parameters = initialize_parameters_deep(layers_dims[0], layers)
#     print(parameters)
#     parameters = initialize_parameters_deep(layers_dims)
#     print(parameters)
    ### END CODE HERE ###
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL, caches = L_model_forward2(X, parameters, layers)
        ### END CODE HERE ###
        
        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(AL, Y)
        ### END CODE HERE ###
    
        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = L_model_backward(AL, Y, caches, layers)
        ### END CODE HERE ###
 
        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if i % 100 == 0:
            costs.append(cost)
        if onIteration is not None and i % 100 == 0:
            onIteration(parameters)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


def L_layer_model_custom(X, Y, layers, learning_rate = 0.0075, num_iterations = 3000, print_cost=False, onIteration = None):
    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
    ### START CODE HERE ###
    parameters = initialize_parameters_deep(X.shape[0], layers)
#     print(parameters)
#     parameters = initialize_parameters_deep(layers_dims)
#     print(parameters)
    ### END CODE HERE ###
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL, caches = L_model_forward2(X, parameters, layers)
        ### END CODE HERE ###
        
        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(AL, Y)
        ### END CODE HERE ###
    
        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = L_model_backward(AL, Y, caches, layers)
        ### END CODE HERE ###
 
        for k in grads:
            a = grads[k]
            where_are_NaNs = np.isnan(a)
            a[where_are_NaNs] = 0
            grads[k] = a
        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###
                
        # Print the cost every 100 training example
        if print_cost and i % 500 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if i % 100 == 0:
            costs.append(cost)
        if onIteration is not None and i % 100 == 0:
            onIteration(parameters)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters
