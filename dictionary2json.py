import sys
import pickle
from collections import defaultdict
import json
from keras.preprocessing.text import Tokenizer


def dict2json(input,output):
    with open(input, "rb") as infile:
        tokenizer = pickle.load(infile)

    with open(output,"w") as outfile:
        json.dump(tokenizer.word_index,outfile)

    infile.close()
    outfile.close()

input = sys.argv[1]
output = sys.argv[2]
dict2json(input, output)
