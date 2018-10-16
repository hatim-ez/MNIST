from numpy import *

def sigmoid(z):

    # SIGMOID returns sigmoid function evaluated at z
    g = zeros(shape(z))

    # ============================= TODO ================================
    # Instructions: Compute sigmoid function evaluated at each value of z.
    g = array([1 / (1 + exp(-x)) for x in z])

    return g
