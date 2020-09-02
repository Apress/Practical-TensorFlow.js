const tf = require('@tensorflow/tfjs-node');
const express = require('express');
const multer = require('multer');

function imageBufferToTensor(imageBuffer) {
  return tf.tidy(() => {
    const tfimage = tf.node.decodeImage(imageBuffer);
    return tfimage.resizeBilinear([224, 224])
      .expandDims()
      .toFloat()
      .div(127)
      .sub(1);
  });
}

async function runServer() {
  const model = await tf.loadLayersModel('file://model/model.json');

  // Building a new model using a truncated version of MobileNet
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
  const cutoffLayer = mobilenet.getLayer('conv_pw_13_relu');
  const truncatedModel = tf.model({ inputs: mobilenet.inputs, outputs: cutoffLayer.output });

  const app = express();
  const storage = multer.memoryStorage();
  const upload = multer({ storage });

  app.post('/upload', upload.single('data'), (req, res) => {
    const img = imageBufferToTensor(req.file.buffer);

    // "Predict" with truncatedModel obtain the activation tensor
    const activation = truncatedModel.predict(img);
    const prediction = model.predict(activation).dataSync();

    res.json({
      prediction,
    });
  });
  app.listen(8081, () => {
    console.log('Ready');
  });
}


runServer();
