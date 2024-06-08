import os
import cv2
import numpy as np
import random
import glob as gb
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import to_categorical
#Data link : https://www.kaggle.com/puneet6060/intel-image-classification

trainpath = '/Users/Saeed/Desktop/deap learing and mchine learning/all_about_machine_and_deep_learning/seg_train/'
testpath = '/Users/Saeed/Desktop/deap learing and mchine learning/all_about_machine_and_deep_learning/seg_test/'
predpath = '/Users/Saeed/Desktop/deap learing and mchine learning/all_about_machine_and_deep_learning/seg_pred/'

code = {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5}

# Function to get the class name from the code
def getcode(n):
    for x, y in code.items():
        if n == y:
            return x

# Image size
s = 100

# Load training data
X_train = []
y_train = []

for folder in os.listdir(trainpath + 'seg_train1'):
    files = gb.glob(pathname=str(trainpath + 'seg_train1/' + folder + '/*.jpg'))
    for file in files:
        image = cv2.imread(file)
        image_array = cv2.resize(image, (s, s))
        image1 = image_array.reshape(-1)
        X_train.append(list(image1))
        y_train.append(code[folder])

# Load test data
X_test = []
y_test = []

for folder in os.listdir(testpath + 'seg_test1'):
    files = gb.glob(pathname=str(testpath + 'seg_test1/' + folder + '/*.jpg'))
    for file in files:
        image = cv2.imread(file)
        image_array = cv2.resize(image, (s, s))
        image1 = image_array.reshape(-1)
        X_test.append(list(image1))
        y_test.append(code[folder])

# Load prediction data
X_pred = []
X_pred_orgnal = []

files = gb.glob(pathname=str(predpath + 'seg_pred1/*.jpg'))
for file in files:
    image = cv2.imread(file)
    image_array = cv2.resize(image, (s, s))
    X_pred_orgnal.append(list(image_array))
    image1 = image_array.reshape(-1)
    X_pred.append(list(image1))

# Convert lists to numpy arrays
X_train = np.array(X_train)
X_pred = np.array(X_pred)
X_pred_orgnal = np.array(X_pred_orgnal)
y_train = np.array(y_train)
y_train = y_train.reshape(len(y_train), 1)
combined_data = np.column_stack((X_train, y_train))

# Shuffle the combined data
np.random.shuffle(combined_data)

# Split the shuffled data back into X_train and y_train
X_train_shuffled = combined_data[:, :-1]  
y_train_shuffled = combined_data[:, -1]  
# Define your neural network class
class NeuralNetwork:
    def __init__(self, input_size, output_size, hidden_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.weights1 = np.random.randn(self.input_size, self.hidden_size)
        self.weights2 = np.random.randn(self.hidden_size, self.output_size)

    def forward(self, x):
        self.z = np.dot(x, self.weights1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.weights2)
        o = self.softmax(self.z3) 
        return o
    
    def sigmoid(self, s):
        return 1.0 / (1 + np.exp(-s))

    def sigmoid_derivative(self, s):
        return s * (1 - s)

    def softmax(self, s):
        exp_s = np.exp(s - np.max(s, axis=1, keepdims=True)) 
        return exp_s / np.sum(exp_s, axis=1, keepdims=True)

    def backward(self, x, y, o, learning_rate):
        self.o_error = y - o  
        self.o_delta = self.o_error * self.sigmoid_derivative(o)
        self.z2_error = self.o_delta.dot(self.weights2.T)
        self.z2_delta = self.z2_error * self.sigmoid_derivative(self.z2)
        self.weights1 += learning_rate * x.T.dot(self.z2_delta)
        self.weights2 += learning_rate * self.z2.T.dot(self.o_delta)

    def train(self, x_train, y_train, learning_rate, epochs, batch_size):
        num_samples = x_train.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))
        for epoch in range(epochs):
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples)
                x_batch = x_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                o = self.forward(x_batch)
                self.backward(x_batch, y_batch, o, learning_rate)

    def predict(self, xpre):
        orre = self.forward(xpre)
        predicted_labels = np.argmax(orre, axis=1)
        return predicted_labels

# Set up your neural network
input_size = 30000   # Flattened image size
output_size = len(code)  # Number of classes
hidden_size = X_train.shape[0]  # Number of neurons in the hidden layer
nn = NeuralNetwork(input_size, output_size, hidden_size)
from sklearn.preprocessing import OneHotEncoder

# Create an instance of the OneHotEncoder
encoder = OneHotEncoder(sparse=False)

# Reshape y_train_shuffled to a column vector
y_train_shuffled = y_train_shuffled.reshape(-1, 1)

# Fit and transform y_train_shuffled to one-hot encoded format
y_train_onehot = encoder.fit_transform(y_train_shuffled)
# Training parameters
learning_rate = 0.09
epochs = 25
batch_size = 32 


nn.train(X_train_shuffled, y_train_onehot, learning_rate, epochs, batch_size)


predicted_labels = nn.predict(X_pred)
print(predicted_labels)

plt.figure(figsize=(20,20))
for n , i in enumerate(range(10)) : 
    plt.subplot(5,5,n+1)
    plt.imshow(X_pred_orgnal[i])
    plt.axis('off')
    plt.title(getcode((predicted_labels[i])))
plt.show()
