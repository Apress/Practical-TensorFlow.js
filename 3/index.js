/*
Demo inspired by https://github.com/ml5js/ml5-examples/blob/development/d3/KMeans/KMeans_GaussianClusterDemo/sketch.js
*/


let model;

// Dataset's path
const csvUrl = 'cluster_df.csv';
// Remote dataset
// csvUrl = 'https://gist.githubusercontent.com/juandes/34d4eb6dfd7217058d56d227eb077ca2/raw/c5c86ea7a32b5ae89ef06734262fa8eff25ee776/cluster_df.csv';


const colMap = {
  0: 'black',
  1: 'green',
  2: 'blue',
  3: 'red',
};

const shapeMap = {
  0: 'circle',
  1: 'square',
  2: 'diamond',
  3: 'cross',
};

async function execute(k) {
  // The k-means model configuration
  const options = {
    k,
    maxIter: 20,
  };

  // Arguments are the file's path, the config, and a callback
  // function that is called once the clustering is finished
  model = ml5.kmeans(csvUrl, options, visualizeResult);
}

function visualizeResult() {
  const x = [];
  const y = [];
  const colors = [];
  const shapes = [];

  model.dataset.forEach((e) => {
    x.push(e[0]);
    y.push(e[1]);
    colors.push(colMap[e.centroid]);
    shapes.push(shapeMap[e.centroid]);
  });

  const trace = {
    x,
    y,
    mode: 'markers',
    type: 'scatter',
    marker: {
      symbol: shapes,
      color: colors,
    },
  };

  Plotly.newPlot('plot', [trace]);
}


function createClusterButton() {
  const btn = document.createElement('BUTTON');
  btn.innerText = 'Cluster!';

  // Listener that waits for clicks.
  // Once a click is done, it will execute the function
  btn.addEventListener('click', () => {
    const slider = document.getElementById('k-range');
    execute(slider.value);
  });

  document.querySelector('#button').appendChild(btn);
}

function init() {
  createClusterButton();
}

init();
