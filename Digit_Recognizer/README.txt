Digit recognizer trained using the MNIST dataset.

Dataset obtained from: https://www.kaggle.com/c/digit-recognizer
CNN model trained using Google Colab.
Accuracy of ~ 99.26 on the test set.
Full code used is in the Digit_Recognizer_MNIST.ipynb notebook.
I have included the training set alone with this: train.csv.zip that needs to be extracted to same directory as the notebook.
test set not included in the zip due to size constraint but can be obtained in the above kaggle link.
mnist_test.csv is the predicted values on the test set.

The model: In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

Required libraries: pandas, numpy, matplotlib, seaborn, keras, sklearn.

The code progresses step by step with comments to show what is being done.

Loading dataset to Colab:
First, to load the dataset directly from kaggle to colab, 
create a kaggle.json file using your username and API key that can be 
obtained at your account in Kaggle and then use 
'!kaggle competitions download -c digit-recognizer' to download the mnist dataset. 
This command is available in the kaggle link above. 
Finally, unzip both the zip files using '!unzip train.csv.zip' 
and '!unzip train.csv.zip' and you're done.






