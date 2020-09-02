let model;
let temperature;

function processInput() {
  const btnInput = document.getElementById('btn-input');
  const pOutput = document.getElementById('p-output');

  // Generate the text after the user clicks the button.
  btnInput.addEventListener('click', () => {
    const text = document.getElementById('input').value;
    model.generate({ seed: text, temperature, length: 100 }, (_, generatedText) => {
      pOutput.innerText = generatedText.sample;
    });
  });
}

function updateSlider() {
  const slider = document.getElementById('temp-range');
  const tempValue = document.getElementById('temp-value');
  tempValue.innerHTML = slider.value;
  temperature = slider.value;

  slider.oninput = function onInputCb() {
    const val = this.value;
    tempValue.innerHTML = val;
    temperature = val;
  };
}

function init() {
  model = ml5.charRNN('models/shakespeare/',
    console.log('Model loaded'));
  updateSlider();
  processInput();
}

init();
