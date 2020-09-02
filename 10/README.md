# Generating Handwritten Digits with Generative Adversarial Networks

## Introduction
For the book's last exercise, we will build the "hello, world" of GANs, a GAN model that generates handwritten digits using the MNIST dataset as the source. Because of the long training time, we will run the training script using Node.js to get as much performance as possible. Then, we are going to write an app that loads the model and produce generated images. Our model's network architecture is based on the Deep Convolutional Generative Adversarial Network (DCGAN) for MNIST example  implemented with Keras (Python) Sequential API.

## Training the model
We will train the model on Node.js. To install the required packages, execute `$ npm i` followed by `$ npm run train` to initialize the training.
The training script will write the model to the *model/* directory. 

## To generate the images
To generate the images, we will use a simple web application with one button that triggers the generation.
To run the app, start a local web server with `$ http-server`, and access the address printed on screen.