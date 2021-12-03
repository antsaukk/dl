import numpy as np
import h5py
from inits import *
from components import *

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False, rs=1):
    """
    Implements a N-layer neural network.
    
    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(rs)
    costs = []                         # keep track of cost
    
    parameters = initialize_parameters_deep(layers_dims, rs)
    
    # training loop
    for i in range(0, num_iterations):

        # forward pass
        AL, caches = L_model_forward(X, parameters)
        
        # cost
        cost = compute_cost(AL, Y)

        # backward propagation
        grads = L_model_backward(AL, Y, caches)
 
        # update params.
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)
    
    return parameters, costs