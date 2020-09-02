const csvUrl = 'https://gist.githubusercontent.com/juandes/2f1ffa32dd4e58f9f5825eca1806244b/raw/c5b387382b162418f051fd83d89fddb4067b91e1/steps_distance_df.csv';
// Local version of the dataset
// const csvUrl = 'steps_distance_df.csv'
const dataSurface = { name: 'Steps and Distance Scatterplot', tab: 'Data' };
const fittedSurface = { name: 'Fitted Dataset', tab: 'Data' };
const dataToVisualize = [];
const predictionsToVisualize = [];

let csvDataset;
let model;

async function defineAndTrainModel(numberEpochs) {
  // Make sure the tfjs-vis visor is open.
  tfvis.visor().open();

  // numOfFeatures is the number of column or features minus the label column
  const numOfFeatures = (await csvDataset.columnNames()).length - 1;

  // Convert the features (xs) and labels (ys) to an array
  const flattenedDataset = csvDataset
    .map(({ xs, ys }) => ({ xs: Object.values(xs), ys: Object.values(ys) }))
    .batch(32);

  // Define the model.
  // Note that there's no activation function because we want
  // a linear relationship
  model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [numOfFeatures],
    units: 1,
  }));

  model.compile({
    optimizer: tf.train.adam(0.1),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'], // Also mean squared error
  });

  // Fit the model using the prepared Dataset
  const history = await model.fitDataset(flattenedDataset, {
    epochs: numberEpochs,
    // Here we want to show on the tfvis visor the loss
    // and mse metric and update it after every epoch.
    callbacks: [
      tfvis.show.fitCallbacks(
        { name: 'Loss and MSE', tab: 'Training' },
        ['loss', 'mse'],
        { callbacks: ['onEpochEnd'] },
      ),
      {
        // Let's also print the loss to the console
        onEpochEnd: async (epoch, logs) => {
          console.log(`${epoch}:${logs.loss}`);
        },
      }],
  });

  // Print the history, model's summary, and weights
  console.log(history);
  drawFittedLine(0, 30000, 500);

  model.summary();
  console.log(`Model weights:\n ${model.getWeights()}`);

  // Get the "predict" button and make it able to perform predictions
  const predictBtn = document.getElementById('predict-btn');
  predictBtn.disabled = false;
}

async function loadData() {
  // Our target variable (what we want to predict) is the the column 'distance'
  // so we add it to the configuration as the label
  csvDataset = tf.data.csv(
    csvUrl, {
      columnConfigs: {
        distance: {
          isLabel: true,
        },
      },
    },
  );


  await csvDataset.forEachAsync((e) => {
    dataToVisualize.push({ x: e.xs.steps, y: e.ys.distance });
  });

  tfvis.render.scatterplot(dataSurface, { values: [dataToVisualize], series: ['Dataset'] });
}

function createLoadPlotButton() {
  const btn = document.createElement('BUTTON');
  btn.innerText = 'Load and plot data';
  btn.id = 'load-plot-btn';

  // Listener that waits for clicks.
  // Once a click is done, it will execute the function
  btn.addEventListener('click', () => {
    loadData();
    const trainBtn = document.getElementById('train-btn');
    trainBtn.disabled = false;
  });


  document.querySelector('#load-plot').appendChild(btn);
}

function createTrainButton() {
  const btn = document.createElement('BUTTON');
  btn.innerText = 'Train!';
  btn.disabled = true;
  btn.id = 'train-btn';

  // Listener that waits for clicks.
  // Once a click is done, it will execute the function
  btn.addEventListener('click', () => {
    const numberEpochs = document.getElementById('number-epochs').value;
    defineAndTrainModel(parseInt(numberEpochs, 10));
  });

  document.querySelector('#train-div').appendChild(btn);
}

function drawFittedLine(min, max, steps) {
  // Empty the array in case the user trains more than once.
  const fittedLinePoints = [];
  const predictors = Array.from(
    { length: (max - min) / steps + 1 },
    (_, i) => min + (i * steps),
  );

  const predictions = model.predict(tf.tensor1d(predictors)).dataSync();

  predictors.forEach((value, i) => {
    fittedLinePoints.push({ x: value, y: predictions[i] });
  });

  const structureToVisualize = {
    values: [dataToVisualize, fittedLinePoints],
    series: ['1. Training Data', '2. Fitted Line'],
  };

  tfvis.render.scatterplot(fittedSurface, structureToVisualize);
}

function createPredictionInput() {
  const input = document.createElement('input');
  input.type = 'number';
  input.id = 'predict-input';

  document.querySelector('#predict').appendChild(input);
}

function createPredictionOutputParagraph() {
  const p = document.createElement('p');
  p.id = 'predict-output-p';

  document.querySelector('#predict').appendChild(p);
}

function createPredictButton() {
  const btn = document.createElement('BUTTON');
  btn.innerText = 'Predict!';
  btn.disabled = true;
  btn.id = 'predict-btn';

  // Listener that waits for clicks.
  // Once a click is done, it will execute the function
  btn.addEventListener('click', () => {
    // Get the value from the input
    const valueToPredict = document.getElementById('predict-input').value;
    const parsedValue = parseInt(valueToPredict, 10);
    const prediction = model.predict(tf.tensor1d([parsedValue])).dataSync();

    // Get the <p> element and append the prediction result
    const p = document.getElementById('predict-output-p');
    p.innerHTML = `Predicted value is: ${prediction}`;

    // Push the input value and the prediction to the predictionsToVisualize array
    // Then, draw it.
    predictionsToVisualize.push({ x: parsedValue, y: prediction });
    const structureToVisualize = {
      values: [dataToVisualize, predictionsToVisualize],
      series: ['1. Training Data', '2. Predictions'],
    };

    tfvis.render.scatterplot(dataSurface, structureToVisualize);
    // Automatically switch to the "Data" tab
    tfvis.visor().setActiveTab('Data');
  });

  document.querySelector('#predict').appendChild(btn);
}

function init() {
  createTrainButton();
  createPredictionInput();
  createPredictButton();
  createPredictionOutputParagraph();
  createLoadPlotButton();
}

init();
