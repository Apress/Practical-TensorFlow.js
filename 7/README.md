# Object detection with a model trained in Google Cloud AutoML

## Introduction
In this chapter, we will create an object detection model. Unlike the past models you created, you will not train this one in TensorFlow.js. On this occasion, you will take the training to the clouds, specifically to Google Cloud, and use their AutoML Vision Object Detection service to prepare the dataset and fit the model. This app loads the model and perform real-time object detection on the browser. Before running it,
make sure the model is available in the directory.

## Running the app
I recommend using npm's http-server. To install it, execute the following instruction `$ npm install http-server -g`.
Then, go to the exercise's root directory, and run `$ http-server`. This command starts the server, and prints the address where you
could access it. By default it is http://127.0.0.1:8080. Then, go to the browser and access that address.

