import nltk
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import pickle
import csv

def process_data(filename):
    list = []
    f = open(filename)
    for row in tqdm(f):
        sentiment = row.split(',')[0]
        print("rr", row.split(',')[0])
        if sentiment == "0":
            sentiment = [1,0,0]
        elif sentiment == "2":
            sentiment = [0,1,0]
        elif sentiment == "4":
            sentiment = [0,0,1]
        print(sentiment)
        list.append([sentiment,row.split(',')[-1]])
    #print(list)




def tokenise(data):
    pass


filename = "testdata.manual.2009.06.14.csv"
process_data(filename)
