// some code to ensure that pressing the ENTER key will
// coresspond to clicking the sumbit button
window.onload = function() {
  // Get the input field
  let input = document.getElementById("userinput");
  // Execute a function when the user releases a key on the keyboard
  input.addEventListener("keyup", function(event) {
    // Cancel the default action, if needed
    event.preventDefault();
    // Number 13 is the "Enter" key on the keyboard
    if (event.keyCode === 13) {
      // Trigger the button element with a click
      document.getElementById("sendinput").click();
    }
  });
};

let model;
// grabs the nn shard and dictionary
function startneuralnetwork() {
  tf.loadModel("./model.json").then(result => {
    model = result;
    // dictionary is already defined in dict.json

    // if the dictionary json location is a url use this code
    // $.getJSON("ENTERURLHERE", function(result) {
    //   dictionary = JSON.parse(result);
    // });
  });
}

function runmodel(input) {
  document.getElementById("sentence").innerHTML = "";
  // when there is no value input into the nn this is the
  // value that pops out (i'd be worried if weren't close to .5)
  let noPrediction = 0.531552;
  let sentence = input.split(" ");
  let answer;
  let outputText = "";
  // Run the neural network against each word in the sentence
  // then display the word along with its sentiment by color
  for (let i = 0; i < sentence.length; i++) {
    let prediction = model.predict(
      tf.tensor2d([sentenceToInput(sentence[i], dictionary)])
    );
    let readable_prediction = prediction.dataSync()[0];
    console.log(
      sentence[i],
      [sentenceToInput(sentence[i] + " ", dictionary)],
      readable_prediction
    );
    if (Math.abs(readable_prediction - noPrediction) <= 0.01) {
      outputText += sentence[i] + " ";
    } else if (readable_prediction >= 0.65) {
      outputText +=
        '<span style="color:green;">' + sentence[i] + "</span>" + " ";
    } else if (readable_prediction > 0.35 && readable_prediction < 0.65) {
      outputText +=
        '<span style="color:black;">' + sentence[i] + "</span>" + " ";
    } else {
      outputText += '<span style="color:red;">' + sentence[i] + "</span>" + " ";
    }
  }
  document.getElementById("sentence").innerHTML = outputText;

  // Run the nn against the whole sentence
  let encoded = sentenceToInput(input, dictionary);
  let prediction = model.predict(tf.tensor2d([encoded]));
  let readable_prediction = prediction.dataSync()[0];
  console.log(sentence, [encoded], readable_prediction);
  if (readable_prediction >= 0.65) {
    answer = "Positive";
  } else if (readable_prediction > 0.35 && readable_prediction < 0.65) {
    answer = "Neutral";
  } else {
    answer = "Negative";
  }
  if (Math.abs(readable_prediction - noPrediction) <= 0.01) {
    gauge.set(0.5);
  } else {
    gauge.set(readable_prediction);
  }

  document.getElementById("answer").innerHTML = answer;
}

// takes the human sentence and converts to an
// array of numbers so the nn can process it
function sentenceToInput(input, dictionary) {
  const maxWords = 16;
  let encoded = [];
  let sentence = input.toLowerCase().split(" ");

  // The neural net was fed in data with padding added to the left
  // so: "The quick brown fox jumps over the lazy dog"
  // is fed in as:
  // [ 0, 0, 0, 0, 0, 0, 0, The, quick, brown, fox, jumps, over, the, lazy, dog]

  // Step 1: add the padding
  let i = 0;
  for (i = 0; i < maxWords - sentence.length; i++) {
    encoded[i] = 0;
  }

  // Step 2: add in the words
  let j = 0;
  for (i = i; i < maxWords; i++) {
    encoded[i] = dictionary[sentence[j]];
    j++;
  }

  // Step 3: iterate over array to ensure there is no nulls
  for (i = 0; i < maxWords; i++) {
    if (encoded[i] == undefined) {
      encoded[i] = 0;
    }
  }
  return encoded;
}

// Gauge needs a global scope but defining
// it as var in buildGauge() is evil
let gauge;

// Gauge Constructor
function buildGauge() {
  let opts = {
    angle: 0.15, // The span of the gauge arc
    lineWidth: 0.44, // The line thickness
    radiusScale: 1, // Relative radius
    pointer: {
      length: 0.6, // // Relative to gauge radius
      strokeWidth: 0.035, // The thickness
      color: "#000000" // Fill color
    },
    limitMax: false, // If false, max value increases automatically if value > maxValue
    limitMin: false, // If true, the min value of the gauge will be fixed
    fontSize: 41,
    colorStart: "#6FADCF", // Colors
    colorStop: "#8FC0DA", // just experiment with them
    strokeColor: "#E0E0E0", // to see which ones work best for you
    generateGradient: true,
    highDpiSupport: true, // High resolution support
    staticZones: [
      { strokeStyle: "#d24c4a", min: 0, max: 0.35 }, // Red from 100 to 130
      { strokeStyle: "#febb35", min: 0.35, max: 0.65 }, // Yellow
      { strokeStyle: "#509b49", min: 0.65, max: 1 } // Green
    ]
  };
  let target = document.getElementById("sentiment");
  gauge = new Gauge(target).setOptions(opts);
  gauge.maxValue = 1;
  gauge.setMinValue(0);
  gauge.animationSpeed = 32;
  gauge.set(0.5);
}

buildGauge();
startneuralnetwork();
