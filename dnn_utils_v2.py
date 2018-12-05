import numpy as np

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def relu_pow(Z, k):
    A = np.maximum(0,Z)
    A = A**k
    
    assert(A.shape == Z.shape)
    
    cache = Z, k
    return A, cache

def relu_pow_backward(dA, cache):
    Z, k = cache
    
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    indexes = Z <= 0
    dZ[indexes] = 0
    
    
    dZ = dZ * k*Z**(k-1)
#     print(Z.shape, dA.shape)
#     assert(False)
    Z0 = Z
    indexes = Z <= 0.1
    Z0[indexes] = 0
    Z1 = Z
    Z1[indexes] = 1
    m = Z.shape[1]
    dk = 1./m * np.sum(Z0**k * np.log(Z1) * dA, axis = 1, keepdims = True)
    if np.isnan(dk).any():
        print(dk)
        print("m", m)
        print("Z0**k", np.sum(Z0**k, axis = 1, keepdims = True))
        print("log", np.sum(np.log(Z1), axis = 1, keepdims = True))
        print(Z0, k, dA)
        assert(False)
    
#     print(type(dk[0][0]))
    assert (dZ.shape == Z.shape)
    assert (dk.shape == k.shape)
    
    return dZ, dk

def relu_leak(Z, k):
#     print(k.shape, Z.shape)
    A = np.array(Z, copy=True)
    
    for y in range(Z.shape[0]):
        for x in range(Z.shape[1]):
            if A[y, x] < 0:
                A[y, x] /= k[y]
#     A[Z < 0] /= k
    
    assert(A.shape == Z.shape)
    
    cache = Z, k
    return A, cache


def relu_leak_backward(dA, cache):
    Z, k = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    dk = np.array(dA * Z, copy=True) # just converting dz to a correct object.
    
    for y in range(Z.shape[0]):
        for x in range(Z.shape[1]):
            if Z[y, x] < 0:
                dZ[y, x] *= k[y]

    
    # When z <= 0, you should set dz to 0 as well. 
#     dZ[Z <= 0] *= k
    dk[Z >= 0] = 0
    dk = np.sum(dk, axis=1, keepdims=True)
    
    assert (dZ.shape == Z.shape)
    assert (dk.shape == k.shape)
    
    return dZ, dk
