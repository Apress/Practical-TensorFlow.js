
// The following "draw" functions are taken from the PoseNet examples from the
// TF.js examples repo:
// https://github.com/tensorflow/tfjs-models/blob/master/posenet/demos/demo_util.js

const COLOR = 'red';

function drawPoint(ctx, y, x, r) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fillStyle = COLOR;
  ctx.fill();
}

function drawSegment([ay, ax], [by, bx], scale, ctx) {
  ctx.beginPath();
  ctx.moveTo(ax * scale, ay * scale);
  ctx.lineTo(bx * scale, by * scale);
  ctx.lineWidth = '10px';
  ctx.strokeStyle = COLOR;
  ctx.stroke();
}

function toTuple({ y, x }) {
  return [y, x];
}

export function drawSkeleton(keypoints, minConfidence, ctx, scale = 1) {
  const adjacentKeyPoints = posenet.getAdjacentKeyPoints(keypoints, minConfidence);

  adjacentKeyPoints.forEach((keypoints) => {
    drawSegment(
      toTuple(keypoints[0].position), toTuple(keypoints[1].position),
      scale, ctx,
    );
  });
}

export function drawKeypoints(keypoints, minConfidence, ctx, scale = 1) {
  for (let i = 0; i < keypoints.length; i += 1) {
    const keypoint = keypoints[i];

    if (keypoint.score >= minConfidence) {
      const { y, x } = keypoint.position;
      drawPoint(ctx, y * scale, x * scale, 3);
    }
  }
}
