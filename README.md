# Image Classification with Simple Neural Network 
Authored by saeed asle

# Description
This project implements an image classification model using a simple neural network in Python.
It utilizes the opencv, numpy, matplotlib, pandas, and keras libraries for image processing, data manipulation, visualization, and model building.

The model is trained on the Intel Image Classification dataset available on Kaggle, which contains images of natural scenes classified into six categories: buildings, forest, glacier, mountain, sea, and street.
The dataset is preprocessed, and images are resized to a standard size before being fed into the neural network for training.

# Features
  * Data loading and preprocessing: Loads image data from directories and preprocesses images for model input.
  * Model architecture: Utilizes a simple neural network with one hidden layer for image classification.
  * Training and evaluation: Trains the model on the dataset and evaluates its performance using accuracy metrics.
  * Prediction: Makes predictions on new images and displays the results with original images.
# How to Use
Ensure you have the necessary libraries installed. You can install them using pip:
    
    pip install opencv-python numpy matplotlib pandas keras
    
  * Download the Intel Image Classification dataset from Kaggle and extract the files.
  * Update the trainpath, testpath, and predpath variables in the code to point to the correct directories where the dataset is stored on your system.
  * Run the provided code to train the model and make predictions on new images.


# Dependencies
  * opencv: For image processing and manipulation.
  * numpy: For numerical operations.
  * matplotlib: For plotting images and training results.
  * pandas: For data manipulation and handling.
  * keras: For building and training the neural network.
# Output
The model outputs predictions for the input images, classifying them into the predefined categories. The code also displays the original images along with their predicted labels for visual verification.
    
