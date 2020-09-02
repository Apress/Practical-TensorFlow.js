import { MnistData } from './data.js';

let model;
let data;
let isModelTrained = false;

let ctx;
const canvasSize = 200;
// Last position of the mouse
let lastPosition = { x: 0, y: 0 };
let drawing = false;

// Image size is [28, 28]
const IMAGE_SIZE = 28;
// 1 because it is a grayscale image
const IMAGE_CHANNELS = 1;

// tfjs-vis visor surface
const dataSurface = { name: 'Sample', tab: 'Data' };

function defineModel() {
  model = tf.sequential();

  // The output of this layer is of shape [24, 24, 8]
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS],
    kernelSize: [5, 5],
    filters: 8,
    strides: 1,
    activation: 'relu',
    padding: 'valid',
  }));

  // The output of this layer is of shape [12, 12, 8]
  model.add(tf.layers.maxPooling2d({
    poolSize: [2, 2],
    strides: 2,
  }));

  // The output of this layer is of shape [8, 8, 16]
  model.add(tf.layers.conv2d({
    kernelSize: [5, 5],
    filters: 16,
    strides: 1,
    activation: 'relu',
    padding: 'valid',
  }));

  // The output of this layer is of shape [4, 4, 16]
  model.add(tf.layers.maxPooling2d({
    poolSize: [2, 2],
    strides: 2,
  }));

  model.add(tf.layers.dropout({ rate: 0.3 }));

  // The output of this layer is of shape [256]
  model.add(tf.layers.flatten());

  // The last layer has 10 outputs; one for each class.
  model.add(tf.layers.dense({
    units: 10,
    activation: 'softmax',
  }));

  model.compile({
    optimizer: tf.train.adam(),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  model.summary();
}

async function train() {
  const BATCH_SIZE = 512;
  const TRAIN_DATA_SIZE = 5500;
  const TEST_DATA_SIZE = 1000;

  // Get a batch of training data
  const [xTrain, yTrain] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    return [
      d.xs.reshape([TRAIN_DATA_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1]),
      d.labels,
    ];
  });

  // Get a batch of test data
  const [xTest, yTest] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    return [
      d.xs.reshape([TEST_DATA_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1]),
      d.labels,
    ];
  });

  await model.fit(xTrain, yTrain, {
    batchSize: BATCH_SIZE,
    epochs: 30,
    validationData: [xTest, yTest],
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Loss and Accuracy', tab: 'Training' },
      ['loss', 'val_loss', 'acc', 'val_acc'],
    ),
  });

  isModelTrained = true;
}


function prepareCanvas() {
  const canvas = document.getElementById('draw-canvas');
  canvas.width = canvasSize;
  canvas.height = canvasSize;
  ctx = canvas.getContext('2d');

  // Set the canvas style
  ctx.strokeStyle = 'white';
  ctx.fillStyle = 'white';
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.lineWidth = 15;

  // Add the canvas event listeners for mouse events
  canvas.addEventListener('mousedown', (e) => {
    drawing = true;
    lastPosition = { x: e.offsetX, y: e.offsetY };
  });

  canvas.addEventListener('mouseout', () => {
    drawing = false;
  });

  canvas.addEventListener('mousemove', (e) => {
    if (!drawing) {
      return;
    }

    ctx.beginPath();
    ctx.moveTo(lastPosition.x, lastPosition.y);
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    lastPosition = { x: e.offsetX, y: e.offsetY };
  });

  canvas.addEventListener('mouseup', () => {
    drawing = false;

    if (!isModelTrained) {
      return;
    }

    // Convert the canvas to a tensor
    // and modify it.
    const toPredict = tf.browser.fromPixels(canvas)
      .resizeBilinear([IMAGE_SIZE, IMAGE_SIZE])
      .mean(2)
      .expandDims()
      .expandDims(3)
      .toFloat()
      .div(255.0);

    const prediction = model.predict(toPredict).dataSync();

    // Use argMax to get the label.
    const p = document
      .getElementById('predict-output');
    p.innerHTML = `Predicted value is: ${tf.argMax(prediction).dataSync()}`;
  });
}

function createButton(innerText, selector, id, listener, disabled = false) {
  const btn = document.createElement('BUTTON');
  btn.innerText = innerText;
  btn.id = id;
  btn.disabled = disabled;

  // Listener that waits for clicks.
  // Once a click is done, it will execute the function.
  btn.addEventListener('click', listener);

  document.querySelector(selector).appendChild(btn);
}

// This function draws a sample of the data
// with tfjs-vis
async function drawData() {
  // Add surface to visor
  const surface = tfvis.visor().surface(dataSurface);
  const results = [];
  const numSamples = 26;
  let digit;

  // Get a sample of N images
  const sample = data.nextTestBatch(numSamples);

  // Create a canvas and add the images
  for (let i = 0; i < numSamples; i += 1) {
    // Cleanup all the allocated tensors
    digit = tf.tidy(() => sample.xs
      .slice([i, 0], [1, sample.xs.shape[1]])
      .reshape([IMAGE_SIZE, IMAGE_SIZE, 1]));

    const visCanvas = document.createElement('canvas');
    visCanvas.width = IMAGE_SIZE;
    visCanvas.height = IMAGE_SIZE;
    visCanvas.style = 'margin: 5px;';
    results.push(tf.browser.toPixels(digit, visCanvas));
    surface.drawArea.appendChild(visCanvas);
  }

  await Promise.all(results);
  digit.dispose();
}

function enableButton(selector) {
  document.getElementById(selector).disabled = false;
}

function init() {
  prepareCanvas();
  createButton('Load data', '#pipeline', 'load-btn',
    async () => {
      data = new MnistData();
      await data.load();
      drawData();
      enableButton('train-btn');
    });

  createButton('Train', '#pipeline', 'train-btn',
    async () => {
      defineModel();
      train();
    }, true);


  createButton('Clear', '#pipeline', 'clear-btn',
    () => {
      ctx.clearRect(0, 0, canvasSize, canvasSize);
    });
}

init();
