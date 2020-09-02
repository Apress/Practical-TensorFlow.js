
let ctx;
const SIZE = 500;

let scoreThreshold = 0.95;
let iouThreshold = 0.5;
let topkThreshold = 10;

// The color of the bounding boxes
const BBCOLOR = '#3498eb';

function setupCamera() {
  const video = document.getElementById('video');
  video.width = SIZE;
  video.height = SIZE;

  navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      width: SIZE,
      height: SIZE,
    },
  }).then((stream) => {
    video.srcObject = stream;
  });

  return new Promise((resolve) => {
    video.onloadedmetadata = () => resolve(video);
  });
}


async function loadVideo() {
  const video = await setupCamera();
  video.play();

  return video;
}

function drawBoundingBoxes(prediction) {
  ctx.font = '20px Arial';
  const {
    left, top, width, height,
  } = prediction.box;

  // Draw the box
  ctx.strokeStyle = BBCOLOR;
  ctx.lineWidth = 1;
  ctx.strokeRect(left, top, width, height);

  // Draw the label background
  ctx.fillStyle = BBCOLOR;
  const textWidth = ctx.measureText(prediction.label).width;
  const textHeight = parseInt(ctx.font, 10);

  // Top left rectangle
  ctx.fillRect(left, top, textWidth + textHeight, textHeight * 2);
  // Bottom left rectangle
  ctx.fillRect(left, top + height - textHeight * 2, textWidth + textHeight, textHeight * 2);

  // Draw labels and score
  ctx.fillStyle = '#000000';
  ctx.fillText(prediction.label, left, top + textHeight);
  ctx.fillText(prediction.score.toFixed(2), left, top + height - textHeight);
}

function detect(model, video) {
  const canvas = document.getElementById('output');
  ctx = canvas.getContext('2d');

  canvas.width = SIZE;
  canvas.height = SIZE;

  async function getBoundingBoxes() {
    const predictions = await model.detect(video, {
      score: scoreThreshold,
      iou: iouThreshold,
      topk: topkThreshold,
    });

    // Modify the video
    ctx.save();
    ctx.scale(-1, 1);
    ctx.translate(-SIZE, 0);
    ctx.drawImage(video, 0, 0, SIZE, SIZE);
    ctx.restore();

    predictions.forEach((prediction) => {
      drawBoundingBoxes(prediction);
    });

    requestAnimationFrame(getBoundingBoxes);
  }

  getBoundingBoxes();
}

function updateSliders(metric, updateAttribute) {
  const slider = document.getElementById(`${metric}-range`);
  const output = document.getElementById(`${metric}-value`);
  output.innerHTML = slider.value;
  updateAttribute(slider.value);

  slider.oninput = function oninputCb() {
    output.innerHTML = this.value;
    updateAttribute(this.value);
  };
}

async function init() {
  const model = await tf.automl.loadObjectDetection('model/model.json');
  const video = await loadVideo();
  detect(model, video);

  updateSliders('score', (value) => {
    scoreThreshold = parseFloat(value);
  });

  updateSliders('iou', (value) => {
    iouThreshold = parseFloat(value);
  });

  updateSliders('topk', (value) => {
    topkThreshold = parseInt(value, 10);
  });
}

init();
