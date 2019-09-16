let $ageInput,
  $ageInputText,
  $fareInput,
  $fareInputText,
  $sexInput,
  $predictBtn,
  $predictionText,
  $status;
let sample = [0, 0, 0];


const TRAIN_DATA_PATH = 'titanic_train.csv';
const TEST_DATA_PATH = 'titanic_test.csv';

let nn;

let data;
const neuralNetOptions = {
  inputs: 3,
  outputs: 2,
  debug: true,
  outputLabels: ['survived'],
  inputLabels: ['age', 'fare', 'is_female'],
  batchSize: 16,
  epochs: 32
}

document.addEventListener('DOMContentLoaded', async function () {

  nn = new NeuralNet(neuralNetOptions)
  initButtons();
  await nn.createModel();
  data = await nn.loadData(TRAIN_DATA_PATH, 'csv');

  normalizedData = await nn.parseData(data)
  await nn.train(normalizedData.inputs, normalizedData.labels);

  
  $status.textContent = "model trained!"
})

function initButtons(){

  $ageInput = document.querySelector('#ageInput');
  $ageInputText = document.querySelector('#ageInputText');
  $fareInput = document.querySelector('#fareInput');
  $fareInputText = document.querySelector('#fareInputText');
  $sexInput = document.querySelector('#sexInput');
  $predictBtn = document.querySelector('#predictBtn');
  $predictionText = document.querySelector('#predictionText');
  $status = document.querySelector('#status');
  
  $status.textContent = "loading data and training model...";

  sample[0] = Number($ageInput.value)
  sample[1] = Number($fareInput.value)
  sample[2] = Number($sexInput.value)

  $ageInput.addEventListener('change', function (e) {
    $ageInputText.textContent = e.target.value
    sample[0] = Number(e.target.value)
  })

  $fareInput.addEventListener('change', function (e) {
    $fareInputText.textContent = e.target.value
    sample[1] = Number(e.target.value)
  })

  $sexInput.addEventListener('change', function (e) {
    sample[2] = Number(e.target.value)
  })
  
  $predictBtn.addEventListener('click', async function(e){
    const result = await nn.predict(sample);
    console.log(result);
    $predictionText.textContent = survivedCode(result);
  })
}


// Returns the string value for Baseball pitch labels
function survivedCode(classNum) {
  switch (classNum) {
      case 0:
          return 'Did not survive';
      case 1:
          return 'Survived!';
      default:
          return 'Unknown';
  }
}
