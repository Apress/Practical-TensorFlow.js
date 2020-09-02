# Building a Linear Regression Model from Scratch

## Introduction
In this exercise, we will keep building on that knowledge and implement a second web application whose primary goal is training a machine learning model. This time, the algorithm in question is linear regression, a supervised learning algorithm, and we will use it to predict the distance I would walk in one day, given the number of taken steps. The web app you will develop through this tutorial follows a similar approach to the previous one. It involves a single activity that downloads a dataset from a given location, processes it, plots it using tfjs-vis, and then fit a model with it. Additionally, unlike the previous application, in this one, you will implement a feature that takes a user's input value to perform a prediction and display on the screen the result.

## Running the app
In this example, you will host the app in a local web server.
A simple one, I recommend is npm's http-server. To install it, execute the following instruction `$ npm install http-server -g`.
Then, go to the exercise's root directory, and run `$ http-server`. This command starts the server, and prints the address where you
could access it. By default it is http://127.0.0.1:8080. Then, go to the browser and access that address.