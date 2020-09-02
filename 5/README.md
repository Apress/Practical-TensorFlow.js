# Making a game with PoseNet, a pose estimator model

## Introduction
In this chapter, you will implement a game that features the model known as PoseNet. PoseNet is a neural network capable of doing real-time human pose estimation in images and videos. In this context, estimating a pose refers to the task of identifying where a person's body joints, and other parts appear in a frame. For example, the figure below is a screenshot from the app, and on the canvas located at the center, you can see a skeleton overlay showing the parts of the body identified by the algorithm. 

The application you will create is a game where the user has to show to PoseNet, through the device's webcam, a series of body parts before time runs out. If PoseNet identifies the part, the user gets one point. Otherwise, it gets a strike.

![screenshot](screenshots/5.1.png)

## Running the app
In this example, you will host the app in a local web server.
I recommend using npm's http-server. To install it, execute the following instruction `$ npm install http-server -g`.
Then, go to the exercise's root directory, and run `$ http-server`. This command starts the server, and prints the address where you
could access it. By default it is http://127.0.0.1:8080. Then, go to the browser and access that address.