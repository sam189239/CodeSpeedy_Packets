Cat Classifier in Python using DL. Dataset included. 
Train a 4 layer model with accuracy of 0.98 on the train set and 0.8 on the test set. 
Includes actual implementation of the network without any framework functions.


Files included:
1) Cat Classifier.py
Main file containing the source code that has to be executed

2) cat_classifier_utils.py
Contains the functions used in the main code and need not be executed.

3) datasets
Folder containing the datasets - train and test.

Required libraries: numpy, h5py, matplotlib

Steps:
First, the data is loaded using the load_data function and the shape of the datasets is printed.
layer_dims constant is declared which has the number of units in each layer of the network. 
The number of layers can be changed by the number of elements in this list - each element is the number of units in each layer. 
The first element is the number of inputs. Since we use images of size 64x64x3 (RGB) the first element is 12288. 
The last element is 1 as it is the output used to classify whether it is a cat or not.
The L_layer_model function is then used to declare the learning model. 
Calling the function runs the training process and prints cost after every 100 iterations.
The function returns the parameters after training which is then used to find the accuracy in the train and test set which is found to be 0.98 and 0.8 respectively.
These paramters can further be used to predict on images of size 64x64.

Reference:
Deep Learning and Neural Networks - deeplearning.ai
