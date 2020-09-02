const tf = require('@tensorflow/tfjs-node');
const data = require('./data');

const IMAGE_SIZE = 28;
const NUM_EPOCHS = 5;
const BATCH_SIZE = 100;
const LATENT_SIZE = 100;

function makeGenerator() {
  const model = tf.sequential();

  model.add(tf.layers.dense({
    inputShape: [LATENT_SIZE],
    units: 7 * 7 * 256,
  }));

  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.leakyReLU());
  model.add(tf.layers.reshape({ targetShape: [7, 7, 256] }));
  model.add(tf.layers.conv2dTranspose({
    filters: 128,
    kernelSize: [5, 5],
    strides: 1,
    padding: 'same',
  }));

  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.leakyReLU());
  model.add(tf.layers.conv2dTranspose({
    filters: 64,
    kernelSize: [5, 5],
    strides: 2,
    padding: 'same',
  }));

  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.leakyReLU());
  model.add(tf.layers.conv2dTranspose({
    filters: 1,
    kernelSize: [5, 5],
    strides: 2,
    padding: 'same',
    activation: 'tanh',
  }));

  return model;
}

function makeDiscriminator() {
  let model = tf.sequential();

  // These are the hidden layers.
  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    filters: 64,
    kernelSize: [5, 5],
    strides: 2,
    padding: 'same',
  }));

  model.add(tf.layers.leakyReLU());
  model.add(tf.layers.dropout(0.3));
  model.add(tf.layers.conv2d({
    filters: 128,
    kernelSize: [5, 5],
    strides: 2,
    padding: 'same',
  }));

  model.add(tf.layers.leakyReLU());
  model.add(tf.layers.dropout(0.3));
  model.add(tf.layers.flatten());
  model.summary();

  // Input and output layers.
  const inputLayer = tf.input({ shape: [IMAGE_SIZE, IMAGE_SIZE, 1] });
  const features = model.apply(inputLayer);
  const outputLayers = tf.layers.dense({ units: 1, activation: 'sigmoid' }).apply(features);

  model = tf.model({ inputs: inputLayer, outputs: outputLayers });
  model.compile({
    optimizer: tf.train.adam(0.0002, 0.5),
    loss: 'binaryCrossentropy',
  });

  return model;
}

function buildCombinedModel(generator, discriminator) {
  const generatorInput = tf.input({ shape: [LATENT_SIZE] });
  const generatorLayers = generator.apply(generatorInput);

  // We want the combined model to only train the generator.
  discriminator.trainable = false;

  const discriminatorLayers = discriminator.apply(generatorLayers);

  const combined = tf.model({ inputs: generatorInput, outputs: discriminatorLayers });
  combined.compile({
    optimizer: tf.train.adam(0.0002, 0.5),
    loss: 'binaryCrossentropy',
  });
  return combined;
}

async function trainDiscriminator(discriminator, generator, xTrain, batchNumber) {
  const [samples, target] = tf.tidy(() => {
    const imageBatch = xTrain.slice(batchNumber * BATCH_SIZE, BATCH_SIZE);
    // tf.randomNormal is the latent space
    const latentVector = tf.randomNormal([BATCH_SIZE, LATENT_SIZE], -1, 1);

    const generatedImages = generator.predict([latentVector], { batchSize: BATCH_SIZE });
    // Mix of real and generated images
    const x = tf.concat([imageBatch, generatedImages], 0);

    // The labels of the imageBatch is 1, and the labels of the generatedImages is 0
    const y = tf.tidy(
      () => tf.concat(
        [tf.ones([BATCH_SIZE, 1]), tf.zeros([BATCH_SIZE, 1])],
      ),
    );
    return [x, y];
  });

  const disLoss = await discriminator.trainOnBatch(samples, target);
  tf.dispose([samples, target]);
  return disLoss;
}

async function trainCombined(combined) {
  const [latent, target] = tf.tidy(() => {
    const latentVector = tf.randomNormal([BATCH_SIZE, LATENT_SIZE], -1, 1);

    // We want the generator labels to be "true" as in not-fake
    // to fool the discriminator
    const trueLabel = tf.tidy(() => tf.ones([BATCH_SIZE, 1]));
    return [latentVector, trueLabel];
  });

  const genLoss = await combined.trainOnBatch(latent, target);
  tf.dispose([latent, target]);
  return genLoss;
}

async function init() {
  const generator = makeGenerator();
  const discriminator = makeDiscriminator();
  const combined = buildCombinedModel(generator, discriminator);


  await data.loadData();
  const { images: xTrain } = data.getData();

  const numBatches = Math.ceil(xTrain.shape[0] / BATCH_SIZE);

  for (let epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
    for (let batch = 0; batch < numBatches; ++batch) {
      const disLoss = await trainDiscriminator(discriminator, generator, xTrain, batch);
      const genLoss = await trainCombined(combined);

      if (batch % 30 === 0) {
        console.log(
          `Epoch: ${epoch + 1}/${NUM_EPOCHS} Batch: ${batch + 1}/${numBatches} - `
          + `Dis. Loss: ${disLoss.toFixed(4)}, Gen. Loss: ${genLoss.toFixed(4)}`,
        );
      }
    }
  }

  await generator.save('file://model/')
    .then(() => console.log('Model saved'));
}

init();
