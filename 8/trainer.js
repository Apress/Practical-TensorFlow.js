const tf = require('@tensorflow/tfjs-node');
// or const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

const NUM_CLASSES = 2;
let xTrain;
let yTrain;

function readImage(path) {
  return tf.tidy(() => {
    const imageBuffer = fs.readFileSync(path);
    const tfimage = tf.node.decodeImage(imageBuffer);
    return tfimage.resizeBilinear([224, 224]) // [244, 244] is the size of MobileNet's training dataset.
      .expandDims()
      .toFloat()
      .div(127.0)
      .sub(1); // This method and .div() bound the values between [-1, 1].
  });
}


async function getImages(dir, label) {
  let img;
  let y;

  fs.readdir(dir, (_, files) => {
    files.forEach(async (file) => {
      img = readImage(`${dir}/${file}`);
      y = tf.tidy(() => tf.oneHot(tf.tensor1d([label]).toInt(), NUM_CLASSES));

      // Add the image and one-hot labels to xTrain and yTrain
      // If xTrain is null, simply set the value.
      if (xTrain == null) {
        xTrain = img;
        yTrain = y;
      } else {
        xTrain = xTrain.concat(img, 0);
        yTrain = yTrain.concat(y, 0);
      }
    });
  });

  tf.dispose(img);
  tf.dispose(y);
}

async function train() {
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
  const cutoffLayer = mobilenet.getLayer('conv_pw_13_relu');

  // Use as inputs everything that comes before cutoffLayer
  // and cutoffLayer as output.
  const truncatedModel = tf.model({
    inputs: mobilenet.inputs,
    outputs: cutoffLayer.output,
  });

  // Create a dataset with the activations (or embeddings)
  // produced by the base model.
  const activation = truncatedModel.predict(xTrain);

  const model = tf.sequential();

  // The inputshape is the shape of truncated's output.
  model.add(tf.layers.flatten(
    { inputShape: truncatedModel.output.shape.slice(1) },
  ));

  model.add(tf.layers.dense({
    units: 20, // Hidden units.
    activation: 'relu',
  }));

  model.add(tf.layers.dense({
    units: NUM_CLASSES,
    activation: 'softmax',
  }));

  // Compile the model
  model.compile({
    loss: 'categoricalCrossentropy',
    optimizer: tf.train.adam(0.001),
    metrics: ['accuracy'],
  });

  await model.fit(activation, yTrain, {
    batchSize: 32,
    epochs: 15,
    callbacks: tf.node.tensorBoard('/tmp/fit_logs'), // Write the logs in the given path.
  });

  await model.save('file://model/');
}

async function init() {
  await getImages('data/pikachu/', 0);
  await getImages('data/bottle/', 1);
  train();
}
init();
