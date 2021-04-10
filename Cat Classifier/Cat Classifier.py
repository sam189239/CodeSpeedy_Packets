import numpy as np
import h5py
import matplotlib.pyplot as plt
import imageio
from cat_classifier_utils import *

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# Example of a picture
index = 10
plt.imshow(train_x_orig[index])
print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")
plt.show()

# Dataset shape
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))

# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T  
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))

# CONSTANTS
layers_dims = [12288, 20, 7, 5, 1] #  4-layer model

#L_layer_model

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    """    
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

    np.random.seed(1)
    costs = []                        
    
    # Parameters initialization. 
    parameters = initialize_parameters_deep(layers_dims)
    
    # Gradient descent
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost.
        cost = compute_cost(AL,Y)
    
        # Backward propagation.
        grads = L_model_backward(AL,Y,caches)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
print('In Train set, ',end='')
pred_train = predict(train_x, train_y, parameters)
print('In Test set, ',end='')
pred_test = predict(test_x, test_y, parameters)

#Try on your own image

image_name = "external-content.duckduckgo.com.jpg" # enter name of the image and make sure it's in the same directory
label_y = [1] # (1 -> cat, 0 -> non-cat)

# image = np.array(plt.imread(image_name))
# image = image.resize(size=(num_px,num_px))
# #np.array(Image.fromarray(image).resize((num_px*num_px*3,1)))
# # my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
# image = image/255.
# my_predicted_image = predict(image, label_y, parameters)

# image_name = "external-content.duckduckgo.com.jpg" # enter name of the image and make sure it's in the same directory
# label_y = [1] # (1 -> cat, 0 -> non-cat)
# im = Image.open(r"external-content.duckduckgo.com.jpg")  
# old_size = im.size
# desired_size = 64
# ratio = float(desired_size)/max(old_size)
# new_size = tuple([int(x*ratio) for x in old_size])
# im = im.resize(new_size, Image.ANTIALIAS)

# new_im = Image.new("RGB", (desired_size, desired_size))
# new_im.paste(im, ((desired_size-new_size[0])//2,
#                     (desired_size-new_size[1])//2))
# plt.imshow(new_im)
# new_im = np.array(new_im)
# np.reshape(new_im, (12288, 1))
# new_im = new_im.flatten()
# new_im = new_im/255.


image = np.array(imageio.imread(image_name))
# my_image = np.array(Image.fromarray(image).resize((64,64)))
my_image = image/255.
my_predicted_image = predict(my_image, label_y, parameters)

plt.show()
print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")