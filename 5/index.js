/* eslint-disable import/prefer-default-export */

import { drawKeypoints, drawSkeleton } from './draw.js';

// Canvas
const SIZE = 500;

// Model
let model;
const MIN_CONFIDENCE = 0.3;

// Gameplay
let timeToPose = 5;
const MAX_STRIKES = 3;
let gameInterval;

// Game status
let inverseMode = false;
let lastPose;
let poseToPerform;
let isInverse;
let score = 0;
let strikes = 0;


const keypointsList = ['nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar', 'leftShoulder',
  'rightShoulder', 'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist', 'leftHip', 'rightHip',
  'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle'];

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

function detect(video) {
  const canvas = document.getElementById('output');
  const ctx = canvas.getContext('2d');
  canvas.width = SIZE;
  canvas.height = SIZE;

  async function getPose() {
    const pose = await model.estimateSinglePose(video, {
      flipHorizontal: true,
    });
    lastPose = pose;


    ctx.clearRect(0, 0, SIZE, SIZE);

    ctx.save();
    // Modify the video
    ctx.scale(-1, 1);
    ctx.translate(-SIZE, 0);
    ctx.drawImage(video, 0, 0, SIZE, SIZE);
    ctx.restore();

    drawKeypoints(pose.keypoints, MIN_CONFIDENCE, ctx);
    drawSkeleton(pose.keypoints, MIN_CONFIDENCE, ctx);

    requestAnimationFrame(getPose);
  }

  getPose();
}


function getRandomInt(max) {
  return Math.floor(Math.random() * Math.floor(max));
}

function nextPose() {
  return [keypointsList[getRandomInt(keypointsList.length)], inverseMode && getRandomInt(3) === 0];
}

function verifyPose() {
  // The default value is inverseMode
  let result = isInverse;
  lastPose.keypoints.forEach((element) => {
    if (poseToPerform === element.part && element.score >= MIN_CONFIDENCE) {
      if (!isInverse) {
        result = true;
      } else {
        result = false;
      }
    }
  });

  return result;
}

function initGame() {
  score = 0;
  strikes = 0;

  [poseToPerform] = nextPose();

  document.getElementById('pose-to-match').innerHTML = `${poseToPerform}`;
}

// This function resets all the labels and the game interval.
function resetGame() {
  ['pose-to-match', 'countdown', 'result-text', 'score', 'strikes'].forEach((id) => {
    document.getElementById(id).innerHTML = '';
  });
  clearInterval(gameInterval);
}

// This function reduces the time to pose to 2 if the score is 3
// and enables the inverse mode of score is 5.
function adjustRules() {
  switch (score) {
    case 3:
      timeToPose = 2;
      break;
    case 5:
      inverseMode = true;
      break;
    default:
      break;
  }
}

function updateState(result) {
  const p = document.getElementById('result-text');
  let text = 'WRONG';
  let textColor = 'RED';
  let gameOver = false;

  switch (result) {
    // If true, increase the counter and tell the user
    // the pose was correct.
    case true:
      score += 1;
      document.getElementById('score').innerHTML = `${score}`;
      text = 'CORRECT';
      textColor = 'GREEN';
      adjustRules();
      break;
    // If false, add a strike.
    case false:
      strikes += 1;
      document.getElementById('strikes').innerHTML = `${strikes}`;

      // End of the game is the player reaches MAX_STRIKES strikes.
      if (strikes >= MAX_STRIKES) {
        text = 'GAME OVER';
        gameOver = true;
        resetGame();
      }

      break;
    default:
      break;
  }

  p.innerHTML = text;
  p.style.color = textColor;

  return gameOver;
}

function initGameLoop() {
  let timeleft = timeToPose;
  initGame();

  gameInterval = setInterval(() => {
    // Verify the pose and update the game state when
    // the counter reaches 0
    if (timeleft < 1) {
      timeleft = timeToPose;
      const result = verifyPose();
      const isGameOver = updateState(result);
      if (isGameOver) {
        return;
      }

      [poseToPerform, isInverse] = nextPose();
      document.getElementById('pose-to-match').innerHTML = `${isInverse ? '~' : ''}${poseToPerform}`;
    }

    document.getElementById('countdown').innerHTML = `${timeleft}`;
    timeleft -= 1;
  }, 1000);
}

function createButton(innerText, selector, id, listener, disabled = false) {
  const btn = document.createElement('BUTTON');
  btn.innerText = innerText;
  btn.id = id;
  btn.disabled = disabled;

  btn.addEventListener('click', listener);
  document.querySelector(selector).appendChild(btn);
}

function prepareButtons() {
  createButton('Start', '#buttons-menu', 'start-btn',
    () => initGameLoop());

  createButton('Stop', '#buttons-menu', 'stop-btn',
    () => resetGame());
}

async function init() {
  const video = await loadVideo();
  model = await posenet.load({
    architecture: 'MobileNetV1',
    outputStride: 16,
    inputResolution: { width: 500, height: 500 },
    multiplier: 0.75,
  });

  prepareButtons();
  detect(video);
}

init();
