from numpy import *
from sigmoid import sigmoid
from costFunction import layerOutput

def predict(Theta, X):
    # Takes as input a number of instances and the network learned variables
    # Returns a vector of the predicted labels for each one of the instances
    
    # Useful values
    m = X.shape[0]
    num_labels = Theta[-1].shape[0]
    num_layers = len(Theta) + 1

    # ================================ TODO ================================
    # You need to return the following variables correctly
    p = []
    for t in range(m):
        layer_output = layerOutput(Theta, X, num_layers, t)
        predicted_label = 0
        max_value = 0
        for i in range(num_labels):
            if layer_output[i] > max_value:
                predicted_label = i
                max_value = layer_output[i]
        p += [predicted_label]
    
    return p

