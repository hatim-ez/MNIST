from numpy import *
from read_dataset import read_dataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from displayData import displayData
from randInitializeWeights import randInitializeWeights
from costFunction import costFunction
from unroll_params import unroll_params
from roll_params import roll_params
from scipy.optimize import *
from predict import predict
from backwards import backwards
from checkNNCost import checkNNCost
from checkNNGradients import checkNNGradients
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient
import datetime as datetime



# ================================ Step 1: Loading and Visualizing Data ================================
print("\nLoading and visualizing Data ...\n")

#Reading of the dataset
# You are free to reduce the number of samples retained for training, in order to reduce the computational cost
size_training = int(input('Please select the training size '))#2000     # number of samples retained for training
size_test     = 10000     # number of samples retained for testing
images_training, labels_training, images_test, labels_test = read_dataset(size_training, size_test)

# Randomly select 100 data points to display
#random_instances = list(range(size_training))
#random.shuffle(random_instances)
#displayData(images_training[random_instances[0:100],:])

#input('Program paused. Press enter to continue!!!')

# ================================ Step 2: Setting up Neural Network Structure &  Initialize NN Parameters ================================
print("\nSetting up Neural Network Structure ...\n")

# Setup the parameters you will use for this exercise
input_layer_size   = 784        # 28x28 Input Images of Digits
num_labels         = 10         # 10 labels, from 0 to 9 (one label for each digit) 

num_of_hidden_layers = int(input('Please select the number of hidden layers: '))
print("\n")

layers = [input_layer_size]
for i in range(num_of_hidden_layers):
    layers = layers +  [int(input('Please select the number nodes for the ' + str(i+1) + ' hidden layers: '))]
layers = layers + [num_labels]

#input('\nProgram paused. Press enter to continue!!!')

print("\nInitializing Neural Network Parameters ...\n")

# ================================ TODO ================================
# Fill the randInitializeWeights.py in order to initialize the neural network weights. 
Theta = randInitializeWeights(layers)

# Unroll parameters
nn_weights = unroll_params(Theta)

#input('\nProgram paused. Press enter to continue!!!')

# ================================ Step 3: Sigmoid  ================================================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.m file.
#

#print("\nEvaluating sigmoid function ...\n")

#g = sigmoid(array([1, -0.5, 0,  0.5, 1]))
#print("Sigmoid evaluated at [1 -0.5 0 0.5 1]:  ")
#print(g)

#input('\nProgram paused. Press enter to continue!!!')

# ================================ Step 4: Sigmoid Gradient ================================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.m file.
#

#print("\nEvaluating Sigmoid Gradient function ...\n")

#g = sigmoidGradient(array([1, -0.5, 0,  0.5, 1]))
#print("Sigmoid Gradient evaluated at [1 -0.5 0 0.5 1]:  ")
#print(g)

#input('\nProgram paused. Press enter to continue!!!')

# ================================ Step 5: Implement Feedforward (Cost Function) ================================

#print("\nChecking Cost Function without Regularization (Feedforward) ...\n")

#lambd = 0.0
#checkNNCost(lambd)

#print('This value should be about 2.09680198349')

#input('\nProgram paused. Press enter to continue!!!')

# ================================ Step 6: Implement Feedforward with Regularization  ================================

#print("\nChecking Cost Function with Reguralization ... \n")

#lambd = 3.0
#checkNNCost(lambd)

#print('This value should be about 2.1433733821')


#input('\nProgram paused. Press enter to continue!!!')

# ================================ Step 7: Implement Backpropagation  ================================

#print("\nChecking Backpropagation without Regularization ...\n")

#lambd = 0.0
#checkNNGradients(lambd)
#input('\nProgram paused. Press enter to continue!!!')


# ================================ Step 8: Implement Backpropagation with Regularization ================================

#print("\nChecking Backpropagation with Regularization ...\n")

#lambd = 3.0
#checkNNGradients(lambd)

#input('\nProgram paused. Press enter to continue!!!')


# ================================ Step 9: Training Neural Networks & Prediction ================================
print("\nTraining Neural Network... \n")

#  You should also try different values of the regularization factor
lambd = float(input('Please select the value for lambda')) #3.0
max_fun = int(input('Please select the value for maxfun')) #3.0

start_training = [datetime.datetime.today().minute, datetime.datetime.today().minute]
res = fmin_l_bfgs_b(costFunction, nn_weights, fprime = backwards, args = (layers,  images_training, labels_training, num_labels, 1.0), maxfun = max_fun, factr = 1., disp = True)
Theta = roll_params(res[0], layers)
end_training = [datetime.datetime.today().minute, datetime.datetime.today().minute]
delta_training = end_training[0] - start_training[0] + (end_training[1] - start_training[1])/60
#input('\nProgram paused. Press enter to continue!!!')
print("Training time :" + str(delta_training) + " minutes")
print("\nTesting Neural Network... \n")

start_testing = [datetime.datetime.today().minute, datetime.datetime.today().minute] #datetime.today()

pred  = predict(Theta, images_test)
end_testing = [datetime.datetime.today().minute, datetime.datetime.today().minute]
delta_testing = end_testing[0] - start_testing[0] + (end_testing[1] - start_testing[1])/60


print('\nLearning :' + str(size_training))
print('\nlambda :' + str(lambd))
print('\nLayers :' + str(num_of_hidden_layers))
print('\nMaxfun :' + str(max_fun))


print('\nTesting time :' + str(delta_testing) + " minutes")

print('\nAccuracy: ' + str(mean(labels_test==pred) * 100))

