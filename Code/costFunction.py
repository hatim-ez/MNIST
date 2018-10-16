from numpy import *
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient
from roll_params import roll_params
from unroll_params import unroll_params

def costFunction(nn_weights, layers, X, y, num_labels, lambd):
    # Computes the cost function of the neural network.
    # nn_weights: Neural network parameters (vector)
    # layers: a list with the number of units per layer.
    # X: a matrix where every row is a training example for a handwritten digit image
    # y: a vector with the labels of each instance
    # num_labels: the number of units in the output layer
    # lambd: regularization factor

    # Setup some useful variables
    m = X.shape[0]
    num_layers = len(layers)

    # Unroll Params
    Theta = roll_params(nn_weights, layers)

    # You need to return the following variables correctly
    J = 0;

    # ================================ TODO ================================
    # The vector y passed into the function is a vector of labels
    # containing values from 1..K. You need to map this vector into a
    # binary vector of 1's and 0's to be used with the neural network
    # cost function.
    yv = zeros((num_labels, m))
    for i in range(m):
        yv[y[i]][i] = 1


    # ================================ TODO ================================
    # In this point calculate the cost of the neural network (feedforward)

    for i in range(m):
        layer_output = layerOutput(Theta, X, num_layers, i)
        for j in range(num_labels):
            cost = -yv[j, i] * log(layer_output[j])
            cost -= (1 - yv[j, i]) * log(1 - layer_output[j])
            J += cost
    J /= m

    # Regularization
    regulation_term = 0
    for i in range(len(Theta)):
        for j in range(Theta[i].shape[0]):
            for k in range(1, Theta[i].shape[1]):
                regulation_term += (Theta[i][j][k]) ** 2
    J += lambd / m * regulation_term / (num_layers - 1 )

    return J



def layerOutput(theta, X, num_layers, i):
    layer_output = append(array([1]), X[i, :])
    for j in range(num_layers-1):  # -1
        layer_output = dot(theta[j], layer_output)
        layer_output = sigmoid(layer_output)
        layer_output = append(array([1]), layer_output)
    return layer_output[1:]