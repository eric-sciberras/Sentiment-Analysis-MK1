import sys
import nltk
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
import pandas as pd
import pickle
from tqdm import tqdm
from collections import defaultdict


# This function is designed to take the raw csv file from the
# sentiment140 dataset and put it in the format:
#   sentiment ::: string
# Where sentiment is a 2x1 array
# [1, 0] == Positive, [0, 1] == negative
def process_raw_data(filename):
    print("Processing Raw Data")
    df = pd.read_csv(filename,header=None,encoding='latin-1')
    df.drop(df.columns[[1,2,3,4]], axis=1, inplace=True)
    df[df.columns[0]] = df[df.columns[0]].map({0:[0], 4: [1]})
    with open("processed_data.csv","w") as output:
        df.to_csv(output,index=False,header=None)
    print("Processing Raw Data: Finished")
    return 0

def data_to_rnn_input(filename, debug=False):

    df = pd.read_csv(filename,header=None)

    text = []
    for x in df[df.columns[1]].values:
        for y in (x.lower()).split(" "):
            text.append(y)

    text = [WordNetLemmatizer().lemmatize(str(word)) for word in tqdm(text)]

    freq_table = defaultdict(int)
    for word in tqdm(text):
            freq_table[word]+=1

    unique_words = set(text)

    # Remove words that rarely appear (since our RNN won't have enough
    # training data for those) and run through the filter
    unique_words = list(filter(lambda x:(freq_table[x]>20), unique_words))
    unique_words = list(filter(lambda x: filter_set(x),unique_words))
    print("Length of input layer: ",len(unique_words))
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(unique_words)
    encoded = tokenizer.texts_to_sequences(df[df.columns[1]])

    if debug:
        reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
        for i in range(0,10):
            print("Original Text: ",df[df.columns[1]][i])
            print("Array: ",encoded[i])
            print("Decoded: ",reverse_dictionary(encoded[i],reverse_word_map))

    # replace text with the encoded arrays
    df[df.columns[1]] = pd.DataFrame({'list':encoded})

    # Save the new csv and the tokeniser
    # (useful for making predicitons with the model)
    with open("processed_data_2.csv","w") as output1:
        df.to_csv(output1,header=False,index=False)

    with open("tokeniser.pickle","wb") as output2:
        pickle.dump(tokenizer,output2)

    print("Processing Finished")
    return 0

def reverse_dictionary(encoded_text,dictionary):
    sentence = ""
    for token in encoded_text:
        sentence+= dictionary[token] + " "
    return sentence

# Mapping function to take a "twitter" sentence and remove elements
# def remove_useless_words(sentence,wordlist):
#     new_sentence = ""
#     for term in nltk.word_tokenize(sentence.lower()):
#         if term in wordlist:
#             new_sentence += term + " "
#     return new_sentence

def filter_set(word):

    ''' Removes words that are:
        - Too short
        - Too long
        - Have symbols
        - Are of a 'useless' type for sentiment analysis e.g.: 'The','To', 'must'
        Link here for the word types:
        https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html '''

    useless_word_types = ['DT','TO','WP','WRB','CC','CD','EX','MD','PRP','IN']
    return(
           word.isalpha() and
           len(word)>2 and
           len(word)<11 and
           str(nltk.pos_tag([word])[0][1]) not in useless_word_types)

filename = sys.argv[1]
processed_data = "processed_data.csv"
process_raw_data(filename)
data_to_rnn_input(processed_data)
