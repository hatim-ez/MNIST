from numpy import *
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient
from roll_params import roll_params
from unroll_params import unroll_params

def backwards(nn_weights, layers, X, y, num_labels, lambd):
    # Computes the gradient fo the neural network.
    # nn_weights: Neural network parameters (vector)
    # layers: a list with the number of units per layer.
    # X: a matrix where every row is a training example for a handwritten digit image
    # y: a vector with the labels of each instance
    # num_labels: the number of units in the output layer
    # lambd: regularization factor

    # Setup some useful variables
    m = X.shape[0]
    num_layers = len(layers)

    # Roll Params
    # The parameters for the neural network are "unrolled" into the vector
    # nn_params and need to be converted back into the weight matrices.
    Theta = roll_params(nn_weights, layers)

    # You need to return the following variables correctly
    Theta_grad = [zeros(w.shape) for w in Theta]

    # ================================ TODO ================================
    # The vector y passed into the function is a vector of labels
    # containing values from 1..K. You need to map this vector into a
    # binary vector of 1's and 0's to be used with the neural network
    # cost function.
    yv = zeros((num_labels, m))
    for i in range(m):
        yv[y[i]][i] = 1


    # ================================ TODO ================================
    # In this point implement the backpropagaition algorithm
    a = [[] for i in range(num_layers)]
    z = [[] for i in range(num_layers)]
    delta=[[] for i in range(num_layers)]
    for t in range(m):
        a[0] = X[t]
        for i in range(0, num_layers - 1):
            a[i] = insert(a[i], 0, 1)
            z[i] = Theta[i].dot(transpose(a[i]))
            a[i + 1] = sigmoid(z[i])
        delta[-1] = a[-1] - yv[:,t]
        for i in range(num_layers - 1, 0, -1):
            if i > 1:
                delta[i - 1] = (transpose(Theta[i-1][:, 1:]).dot(delta[i])) * sigmoidGradient(z[i - 2]) #because z[0] corresponds to z2

        for i in range(0, num_layers - 1):
            Theta_grad[i] += atleast_2d(delta[i+1]).T.dot(atleast_2d(a[i]))

    # regularization
    for l in range(0, num_layers - 1):
        for i in range(Theta[l].shape[0]):
            for j in range(1, Theta[l].shape[1]):
                Theta_grad[l][i][j] += lambd * Theta[l][i][j]

    # Unroll Params
    Theta_grad = unroll_params(Theta_grad)

    return Theta_grad/m


