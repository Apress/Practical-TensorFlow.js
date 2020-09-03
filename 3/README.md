# Doing k-means with ml5.js and Plotly

## Introduction
For our next exercise, we will leave behind the topic of supervised learning, and consider its counterpart, unsupervised learning, 
and its quintessential algorithm, k-means.

In this section, you will write a web app that fits a k-means model using a synthetic dataset while visualizing its different outcomes using the visualization library Plotly. Contrary to the previous two problems where
you implemented the algorithms from scratch, in this one, you will use an external library named ml5.js, a higher-level abstraction of TensorFlow.js that provides a simple API with several prebuilt and ready-to-use models.

## Running the app
In this example, you will host the app in a local web server.
I recommend using npm's http-server. To install it, execute the following instruction `$ npm install http-server -g`.
Then, go to the exercise's root directory, and run `$ http-server`. This command starts the server, and prints the address where you
could access it. By default it is http://127.0.0.1:8080. Then, go to the browser and access that address.

