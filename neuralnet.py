import ast
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import pandas as pd
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
import pickle



def neuralnet(filename):
    df = pd.read_csv(filename,header=None)

    ###### WARNING ######
    # ast.literal_eval() is a DANGEROUS function since it can result in arbitary
    # code execution. I may try to remove code later down the track, but since
    # we are dealing with data that doesn't contain code it is safe.
    # TODO: Change from csv to pickle
    X = [ ast.literal_eval(x) for x in df[df.columns[1]].values ]
    Y = [ ast.literal_eval(x) for x in df[df.columns[0]].values ]

    training_size = 0.8
    max_words = 16

    X_train = sequence.pad_sequences( X[:int(training_size*len(X))], maxlen=max_words)
    X_test =  sequence.pad_sequences( X[-int((1-training_size)*len(X)):], maxlen=max_words)

    Y_train = sequence.pad_sequences( Y[:int(training_size*len(Y))], maxlen=2)
    Y_test =  sequence.pad_sequences( Y[-int((1-training_size)*len(Y)):], maxlen=2)

    embedding_size=32
    model=Sequential()
    model.add(Embedding(len(X_train) + len(X_test), embedding_size, input_length=max_words))
    model.add(LSTM(100))
    model.add(Dense(2, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    batch_size = 100
    num_epochs = 30

    X_valid, y_valid = X_train[:batch_size], Y_train[:batch_size]
    X_train2, y_train2 = X_train[batch_size:], Y_train[batch_size:]

    filepath="sentiment-ai-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]


    model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list, verbose=0)

    model.save('my_model.h5')
    return 0

    '''
    print("X_train: ", len(X_train))
    print("Y_train: ", len(Y_train))
    print("X_test: ", len(X_test))
    print("Y_test: ", len(Y_test))
    print(len(X_train) + len(X_test))
    '''

def play_with_model():
    max_words = 16

    with open( "tokeniser.pickle", "rb" ) as input1:
        tokenizer = pickle.load(input1)

    with open( "model.pickle", "rb" ) as input2:
        model = pickle.load(input2)

    sentence = [sys.argv[1]]
    encoded = tokenizer.texts_to_sequences(sentence)
    print(encoded)
    encoded = sequence.pad_sequences(encoded, maxlen=max_words)

    #print("Original Text", sys.argv[1])
    #print("encoded: ",encoded)
    #print("Prediction: ",model.predict(encoded))
    # while True:
    #     inp = input('Enter A Sentence: ')
    #     encoded = tokenizer.texts_to_sequences(inp)
    prediction = model.predict(encoded).tolist()

    max_index = prediction.index(max(prediction))
    if max_index == 0:
        print("Positive")
    elif max_index == 1:
        print("Neutral")
    elif max_index == 2:
        print("Negative")
    else:
        print("ERROR")
    return 0

neuralnet("processed_data_2.csv")
play_with_model()