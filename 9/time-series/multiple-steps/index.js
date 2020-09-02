const csvUrl = 'https://gist.githubusercontent.com/juandes/5c7397a2b8844fbbbb2434011e9d9cc5/raw/9a849143a3e3cb80dfeef3b1b42597cc5572f674/sequence.csv';
// const csvUrl = 'data/sequence.csv';
const testUrl = 'https://gist.githubusercontent.com/juandes/950003d00bd16657228e4cdd268a312a/raw/e5b5d052f95765d5bedfc6618e3c47c711d6816d/test.csv';
// const testUrl = 'data/test.csv';


const TIMESTEPS = 60;
const FUTURE_TIMESTEPS = 55;
const TRAINING_DATASET_SIZE = 900;
const TEST_DATASET_SIZE = 8;

let model;

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


async function defineAndTrain(xTrain, yTrain) {
  model = tf.sequential();
  model.add(tf.layers.lstm(
    {
      inputShape: xTrain.shape.slice(1),
      units: 16,
      returnSequences: true,
    },
  ));
  model.add(tf.layers.lstm(
    {
      units: 8,
      activation: 'relu',
    },
  ));
  model.add(tf.layers.dense({ units: yTrain.shape[1] }));
  model.compile({
    loss: 'meanSquaredError',
    optimizer: tf.train.adam(0.01),
  });

  await model.fit(xTrain, yTrain, {
    batchSize: 32,
    epochs: 15,
    callbacks: [
      tfvis.show.fitCallbacks(
        document.getElementById('canvas-training-tfvis'),
        ['loss'],
        { callbacks: ['onEpochEnd', 'onBatchEnd'] },
      )],
  });
}

function predict(xTest) {
  return model
    .predict(xTest).arraySync();
}

function prepareMultiDataset(trainingSet, testingSet) {
  let [xTrain, yTrain] = tf.split(trainingSet.xs, [FUTURE_TIMESTEPS, TIMESTEPS - FUTURE_TIMESTEPS], 1);
  yTrain = yTrain.squeeze(2);
  const [xTest, yTest] = tf.split(testingSet.xs, [FUTURE_TIMESTEPS, TIMESTEPS - FUTURE_TIMESTEPS], 1);

  return {
    xTrain, yTrain, xTest, yTest,
  };
}

async function prepareDataset(dataset, size) {
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


function range(min, max, steps) {
  return Array.from({ length: (max - min) / steps + 1 }, (_, i) => min + i * steps);
}

async function plotPrediction(which, xTest, yTest, predictions) {
  let testCase = (await xTest.array());
  let targets = (await yTest.array());
  testCase = testCase[which].flat();
  targets = targets[which].flat();

  const traceSequence = {
    x: range(0, FUTURE_TIMESTEPS, 1),
    y: testCase,
    mode: 'lines',
    type: 'scatter',
    name: 'Test data',
    line: {
      width: 3,
    },
  };

  const traceActualValue = {
    x: range(FUTURE_TIMESTEPS, TIMESTEPS - 1, 1),
    y: targets,
    mode: 'lines',
    type: 'scatter',
    name: 'Actual value',
    line: {
      dash: 'dash',
      width: 3,
    },
  };

  const tracePredictedValue = {
    x: range(FUTURE_TIMESTEPS, TIMESTEPS - 1, 1),
    y: predictions[which],
    mode: 'lines',
    type: 'scatter',
    name: 'Predicted value',
    line: {
      dash: 'dashdot',
      width: 3,
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
  const trainingSet = await prepareDataset(trainingDataset, TRAINING_DATASET_SIZE);
  const testingSet = await prepareDataset(testDataset, TEST_DATASET_SIZE);

  const {
    xTrain, yTrain, xTest, yTest,
  } = prepareMultiDataset(trainingSet, testingSet);

  const testCasesIndex = range(1, 8, 1);
  testCasesIndex.forEach((testCase) => {
    createButton(`Test case ${testCase}`, '#test-buttons', `test-case-${testCase}`,
      async () => {
        plotPrediction(testCase - 1, xTest, yTest, predictions);
      }, true);
  });

  const trainButton = document.getElementById('btn-train');
  trainButton.addEventListener('click', async () => {
    await defineAndTrain(xTrain, yTrain);
    predictions = predict(xTest);
    testCasesIndex.forEach((testCase) => {
      document.getElementById(`test-case-${testCase}`).disabled = false;
    });
  });
}

init();
