# Flowers Recognition using Convolusion Neural Network
This project uses the Convolusion Neural Network(CNN) model to predict the image given by flower recognition dataset on kaggle.


Kaggle – Flowers Recognition  (url = https://www.kaggle.com/alxmamaev/flowers-recognition)

There ara 5 categories of flowers that are predicted and trained on:
* Daisy
* Dandelion
* Rose
* Sunflower
* Tulip

There are labeled 4242 images of flowers.

The model has three convolutional layers. Categorical_crossentropy is used for loss and Adam is used as the optimizer. In addition, this project will output a visual prediction on ten random images in the test set and make their confusion matrix.

### **HOW**
  1.	Read Folder
  2.	Image is processed as follows: resize, storing the category of each image
  3.	split the dataset into training set and test set
  4.	Create a keras CNN model (Sequential model)
  5.	Do Data Augmentation(image rotation, shift, flip, zoom), increase training samples, and enhance the recognition rate of CNN
  6.	Compiling the CNN model
  7.	Batch Training
  8.	Compare the results of ten random predictions in the test set with the correct answers and visualize the output
  9.	Make Confusion matrix
  
### **Confusion Matrix**
The numbers in the Output_plot.png/Confusion Matrix represent：

>0 = Daisy  
>1 = Dandelion  
>2 = Rose  
>3 = Sunflower  
>4 = Tulip  
  

**! Please note that you will need to change the file been used in the code to wherever you have stored the dataset.**
