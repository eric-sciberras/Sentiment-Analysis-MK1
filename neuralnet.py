import ast
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import pandas as pd
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
import pickle
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import sys
from tqdm import tqdm

def neuralnet(filename):
    df = pd.read_csv(filename,header=None)

    with open( "tokeniser.pickle", "rb" ) as input1:
        tokenizer = pickle.load(input1)

    ###### WARNING ######
    # ast.literal_eval() is a DANGEROUS function since it can result in arbitary
    # code execution. I may try to remove code later down the track, but since
    # we are dealing with data that doesn't contain code it is safe.
    # TODO: Change from csv to pickle
    X = [ ast.literal_eval(x) for x in tqdm(df[df.columns[1]].values) ]
    Y = [ ast.literal_eval(x) for x in tqdm(df[df.columns[0]].values) ]

    training_size = 0.8
    max_words = 16

    X_train = sequence.pad_sequences( X[:int(training_size*len(X))], maxlen=max_words)
    X_test =  sequence.pad_sequences( X[-int((1-training_size)*len(X)):], maxlen=max_words)

    Y_train = sequence.pad_sequences( Y[:int(training_size*len(Y))], maxlen=1)
    Y_test =  sequence.pad_sequences( Y[-int((1-training_size)*len(Y)):], maxlen=1)

    embedding_size=32
    model=Sequential()
    model.add(Embedding(len(tokenizer.word_index.items())+1, embedding_size, input_length=max_words))
    #model.add(Embedding(int((len(X_train) + len(X_test))/1.2), embedding_size, input_length=max_words))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    batch_size = 300
    num_epochs = 20

    X_valid, y_valid = X_train[:batch_size], Y_train[:batch_size]
    X_train2, y_train2 = X_train[batch_size:], Y_train[batch_size:]

    filepath="sentiment-ai-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
    callbacks_list = [checkpoint]

    model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list)

    model.save("my-model.hdf5")
    return 0

neuralnet("processed_data_2.csv")
