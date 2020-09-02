# Recognizing handwritten digits with convolutional neural networks

## Introduction
For this chapter's exercise, you will build an app that fits a convolutional neural network using the MNIST
dataset to identify in real-time handwritten digits. At first, as we did with the others, we will load the dataset,
design the model, and train it. Then, we will develop the drawing functionality. This feature involves setting up a
canvas where the user will draw a digit using its mouse. After drawing the digit, the app converts the image to a tensor
and feeds it to the model to recognize the input value.

## Running the app
In this example, you will host the app in a local web server.
I recommend using npm's http-server. To install it, execute the following instruction `$ npm install http-server -g`.
Then, go to the exercise's root directory, and run `$ http-server`. This command starts the server, and prints the address where you
could access it. By default it is http://127.0.0.1:8080. Then, go to the browser and access that address.