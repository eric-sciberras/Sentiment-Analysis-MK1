# Sentiment-Analysis-MK1
A sentiment analysis project based on [Sentiment140](http://help.sentiment140.com/for-students) training data. This First attempt utilises some basic rules for creating quality input data from the raw data alongside a LSTM neural network.

## Getting Started

### Play With Already Trained Neural Network

1. Go to `html` folder and and open `nn.html`

### Train Neural Network Yourself

1. Download the sentiment140 data from [here](http://help.sentiment140.com/for-students). Alternatively you may use your own dataset however the `process_raw_data()` function may need to be altered so it is in the correct format.

2. Run `pip3 install -r requirements.txt` to install all the python3 libraries required for use. 

3. Run `python3 processData /path/to/dataset/'` to create 3 files:
   * `processed_data.csv` : which is the data in correct format for converting sentences to arrays
   * `processed_data_2.csv` : which is data that is ready to be fed into the neural network
   * `tokeniser.pickle` : holds the dictionary which is useful for converting the words to numbers and vice versa
 
4. Run `python3 neuralnet` to start training. Note: it is recommended to use tensorflow-gpu for this as running on a cpu will be very slow. This will create a .hdf5 file for each epoch e.g. sentiment-ai-04-0.74. The format is sentiment-ai-<epoch>-<val_acc> and a file called `my-model.hdf5` which is the last epoch. 

5. To play with the model:
   + Replace your `model.hdf5` with the one contained in `html` folder and open `nn.html` (Recommended) OR
   + Run `python3 runmodel /path/to/model.hdf5` 

## Possible Improvements For MK-2

+ Use more or better datasets (create my own ???)
+ Research Word2vec and its possible advantages 
+ Allow neural network to accept arbitrarily sized sentences
+ Use more sophisticated techniques for processing raw datasets (i.e: identifying words that don't contribute to sentiment)
+ Optimise parameters for LSTM neural network or use different neural network

