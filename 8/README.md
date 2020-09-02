# Training an image classifier with transfer learning on Node.js

## Introduction
In this exercise, you will train an image classifier using transfer learning.
To deploy, you will write a Node.js backend application that loads and serves the model
using an web service.

## Running the app
These applications require Node.js. To install it, visit [Node.js](https://nodejs.org/en/).
To run the Trainer app, execute `$ npm run train`. To serve it, run `npm run serve`.
Once the server is up, you can test it with a command like this `$ curl -F "data=@data/{DIR_NAME}/{IMAGE_NAME}" http://localhost:8081/upload`, where
`data` is the path to an image.


