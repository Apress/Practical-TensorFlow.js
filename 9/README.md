# Time Series Forecasting and Text Generation with Recurrent Neural Networks

## Introduction
This chapter consists of two parts. In the first one, we will go through the steps of creating and training two recurrent neural networks to do time series forecasting using the steps dataset introduced earlier in the book. The first RNN forecasts a single point in the future, while the second, forecasts multiple steps. Then, we will create a second app that loads an ml5.js' RNN trained on a corpus of Shakespeare's works to generate "Shakespeare-like" texts.

## Running the app
To run the applications, use npm's *http-server* to start a local web server.
To install it, execute the following instruction `$ npm install http-server -g`.
Then, go to the exercise's root directory, and run `$ http-server`. This command starts the server, and prints the address where you
could access it. By default it is http://127.0.0.1:8080. Then, go to the browser and access that address.

