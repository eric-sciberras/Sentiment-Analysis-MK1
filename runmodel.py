import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import pickle
from keras.models import load_model
import sys

def play_with_model(model):
    max_words = 16
    with open( "tokeniser.pickle", "rb" ) as input1:
        tokenizer = pickle.load(input1)
    model = load_model(model)
    while True:
        sentence = [input("Enter Sentence: ")]
        encoded = tokenizer.texts_to_sequences(sentence)
        print(encoded)
        encoded = sequence.pad_sequences(encoded, maxlen=max_words)

        prediction = model.predict(encoded).tolist()

        print("Prediction: ",prediction)
        value = float(prediction[0][0])
        print(value)
        if value >= 0.5:
            print("Positive")
        elif value < 0.5:
            print("Negative")
        else:
            print("ERROR")

model = sys.argv[1]
play_with_model(model)
