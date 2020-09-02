/**
 * In this web app, we'll fit a logistic regression model to predict the outcome of a toy dataset.
 */

// Dataset's url
const csvUrl = 'https://gist.githubusercontent.com/juandes/ba58ef99df9bd719f87f807e24f7ea1c/raw/59f57af034c52bd838c513563a3e547b3650e7ba/lr-dataset.csv';
let dataset;

async function defineAndTrainModel(numberEpochs) {
  // numOfFeatures is the number of column or features minus the label column
  const numOfFeatures = (await dataset.columnNames()).length - 1;
  const trainingSurface = { name: 'Loss and MSE', tab: 'Training' };

  const flattenedDataset = dataset
    .map(({ xs, ys }) => ({ xs: Object.values(xs), ys: Object.values(ys) }))
    // Convert the features (xs) and labels (ys) to an array
    .batch(10)
    .shuffle(100, 17); // buffer size and seed

  // Define the model.
  const model = tf.sequential();

  // Add a Dense layer to the Sequential model
  model.add(tf.layers.dense({
    inputShape: [numOfFeatures],
    units: 1,
    activation: 'sigmoid',
  }));

  model.compile({
    optimizer: tf.train.adam(0.1),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });

  // Print the summary to console
  model.summary();


  // Fit the model
  await model.fitDataset(flattenedDataset, {
    epochs: numberEpochs,
    callbacks: [
      // Show on a tfjs-vis visor the loss and accuracy values at the end of each epoch.
      tfvis.show.fitCallbacks(
        trainingSurface,
        ['loss', 'acc'],
        { callbacks: ['onEpochEnd'] },
      ),
      {
        // Print to console the loss value at the end of each epoch.
        onEpochEnd: async (epoch, logs) => {
          console.log(`${epoch}:${logs.loss}`);
        },
      },
      {
        onTrainEnd: async () => {
          console.log('Training has ended.');
        },
      }],
  });

  // Output value should be near 0.
  model.predict(tf.tensor2d([[0.1773208878849, -1.447465411302]])).print();
  // Output value should be near 1.
  model.predict(tf.tensor2d([[-1.58566906881, 1.91762229933]])).print();
}

function loadData() {
  // Our target variable (what we want to predict) is the column 'label' (wow, very original),
  // so we specify it in the configuration object as the label
  dataset = tf.data.csv(
    csvUrl, {
      columnConfigs: {
        label: {
          isLabel: true,
        },
      },
    },
  );
}

async function visualizeDataset() {
  // tfjs-vis surface's names and tabs
  const dataSurface = { name: 'Scatterplot', tab: 'Charts' };
  const classZero = [];
  const classOne = [];

  await dataset.forEachAsync((e) => {
    // Extract the features from the dataset
    const features = { x: e.xs.feature_1, y: e.xs.feature_2 };
    // If the label is 0, add the features to the "classZero" array
    if (e.ys.label === 0) {
      classZero.push(features);
    } else {
      classOne.push(features);
    }
  });

  // Specify the name of the labels. This will show up in the chart's legend.
  const series = ['Class 0', 'Class 1'];
  const dataToDraw = { values: [classZero, classOne], series };

  tfvis.render.scatterplot(dataSurface, dataToDraw, {
    xLabel: 'feature_1',
    yLabel: 'feature_2',
    zoomToFit: true,
  });
}

function createTrainButton() {
  const btn = document.createElement('BUTTON');
  btn.innerText = 'Train!';

  // Listener that waits for clicks.
  // Once a click is done, it will execute the function
  btn.addEventListener('click', () => {
    const numberEpochs = document.getElementById('number-epochs').value;
    console.log(numberEpochs);
    defineAndTrainModel(parseInt(numberEpochs, 10));
  });

  document.querySelector('#train').appendChild(btn);
}

function createVisualizeButton() {
  const btn = document.createElement('BUTTON');
  btn.innerText = 'Visualize!';

  // Listener that waits for clicks.
  // Once the button is clicked, it will call "visualizeDataset"
  btn.addEventListener('click', () => {
    visualizeDataset();
  });

  // the # means ID, so it is searching with the element with id visualize
  document.querySelector('#visualize').appendChild(btn);
}

function init() {
  createTrainButton();
  createVisualizeButton();
  loadData();
}

init();
