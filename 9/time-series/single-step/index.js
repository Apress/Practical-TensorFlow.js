// const csvUrl = 'https://gist.githubusercontent.com/juandes/5c7397a2b8844fbbbb2434011e9d9cc5/raw/9a849143a3e3cb80dfeef3b1b42597cc5572f674/sequence.csv';
const csvUrl = 'data/sequence.csv';
// const testUrl = 'https://gist.githubusercontent.com/juandes/950003d00bd16657228e4cdd268a312a/raw/e5b5d052f95765d5bedfc6618e3c47c711d6816d/test.csv';
const testUrl = 'data/test.csv';


const TIMESTEPS = 60;
const TRAINING_DATASET_SIZE = 900;
const TEST_DATASET_SIZE = 8;

let model;

// Load the training and test data
function loadData() {
  const trainingDataset = tf.data.csv(csvUrl, {
    columnConfigs: {
      value: {
        isLabel: true,
      },
    },
  });

  const testDataset = tf.data.csv(testUrl, {
    columnConfigs: {
      value: {
        isLabel: true,
      },
    },
  });

  return { trainingDataset, testDataset };
}

async function defineModel() {
  model = tf.sequential();

  model.add(tf.layers.lstm(
    {
      inputShape: [TIMESTEPS, 1],
      units: 32,
      returnSequences: false,
    },
  ));
  model.add(tf.layers.dense({ units: 1 }));

  model.compile({
    loss: 'meanSquaredError',
    optimizer: tf.train.adam(0.1),
  });
}

async function trainModel(trainingSet) {
  // Get the container to draw the tfjs-vis plots.
  const container = document.getElementById('canvas-training-tfvis');

  await model.fit(trainingSet.xs, trainingSet.ys, {
    batchSize: 64,
    epochs: 25,
    validationSplit: 0.1,
    callbacks: [
      tfvis.show.fitCallbacks(
        container,
        ['loss', 'val_loss'],
        { callbacks: ['onEpochEnd', 'onBatchEnd'] },
      )],
  });
}

async function prepareDataset(dataset, size) {
  // Create two TensorBuffers to add the sequences and targets.
  const sequences = tf.buffer([size, TIMESTEPS, 1]);
  const targets = tf.buffer([size, 1]);

  let row = 0;
  await dataset.forEachAsync(({ xs, ys }) => {
    let column = 0;
    Object.values(xs).forEach((element) => {
      sequences.set(element, row, column, 0);
      column += 1;
    });
    targets.set(ys.value, row, 0);
    row += 1;
  });

  return { xs: sequences.toTensor(), ys: targets.toTensor() };
}

function predict(testingSet) {
  return model
    .predict(testingSet.xs)
    .dataSync();
}

// range is a "range-like" function that produces an array
// of values from min to max.
function range(min, max, steps) {
  return Array.from({ length: (max - min) / steps + 1 }, (_, i) => min + i * steps);
}

async function plotPrediction(which, testingSet, predictions) {
  let testCase = (await testingSet.xs.array());
  testCase = testCase[which].flat();

  // These are the Plotly traces to draw the test examples.
  // The first trace draws the test data.
  const traceSequence = {
    x: range(0, TIMESTEPS - 1, 1),
    y: testCase.slice(0, TIMESTEPS),
    mode: 'lines',
    type: 'scatter',
    name: 'Test data',
  };

  // The second trace draws the actual value.
  const traceActualValue = {
    x: [TIMESTEPS],
    y: [testCase[TIMESTEPS - 1]],
    mode: 'markers',
    type: 'scatter',
    name: 'Actual value',
    marker: {
      symbol: 'circle',
    },
  };

  // The second trace draws the predicted value.
  const tracePredictedValue = {
    x: [TIMESTEPS],
    y: [predictions[which]],
    mode: 'markers',
    type: 'scatter',
    name: 'Predicted value',
    marker: {
      symbol: 'diamond',
    },
  };

  const traces = [traceSequence, traceActualValue, tracePredictedValue];
  Plotly.newPlot('plot', traces);
}

function createButton(innerText, selector, id, listener, disabled = false) {
  const btn = document.createElement('BUTTON');
  btn.innerText = innerText;
  btn.id = id;
  btn.disabled = disabled;

  btn.addEventListener('click', listener);
  document.querySelector(selector).appendChild(btn);
}


async function init() {
  let predictions;
  const { trainingDataset, testDataset } = loadData();
  const train = await prepareDataset(trainingDataset, TRAINING_DATASET_SIZE);
  const test = await prepareDataset(testDataset, TEST_DATASET_SIZE);

  const testCasesIndex = range(1, 8, 1);

  // Create the buttons for the test cases.
  // Pressing one of the buttons, draw the test case and prediction.
  testCasesIndex.forEach((testCase) => {
    createButton(`Test case ${testCase}`, '#test-buttons', `test-case-${testCase}`,
      async () => {
        plotPrediction(testCase - 1, test, predictions);
      }, true);
  });

  const trainButton = document.getElementById('btn-train');

  trainButton.addEventListener('click', async () => {
    await defineModel();
    await trainModel(train);
    predictions = predict(test);

    testCasesIndex.forEach((testCase) => {
      document.getElementById(`test-case-${testCase}`).disabled = false;
    });
  });
}


init();
