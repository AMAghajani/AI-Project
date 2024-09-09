from template import NaiveBayesClassifier
import pandas as pd
import re
from unidecode import unidecode
from nltk.stem import PorterStemmer
from time import time

def preprocess(tweet_string):
    # clean the data and tokenize it
    str = re.sub(r"[&!\"#$%&()*+-./:;<=>?@[\]^_{|}~\n -' 0123456789\\]"," ", tweet_string)
    str = unidecode(str)
    strs = str.split(' ')
    stemmer = PorterStemmer()
    features = list(stemmer.stem(word) for word in strs)
    features = list(filter(None , features))
    li = ["to", "of" , "as" , "a" , "an" , "with" , "for" , "and" , "or" , "my", "your" , "his", "her", "ours", "their", "me", "him", "us" , "them", "i" , "you" , "he" , "she" , "we" , "they" , "am" , "is" , "are", "the", "at", "it", "its", "im", "ill", "in", "from"]
    final_features = []
    for word in features:
        if not (word in li or word.isnumeric()) and len(word) > 1:
            final_features.append(word)
    return final_features


def load_data(data_path):
    # load the train & evaluation csv file and return the data
    grass_data = pd.read_csv(data_path)
    data = []
    for tweet in grass_data.iterrows():
        data.append((preprocess(str(tweet[1].text)), str(tweet[1].label_text)))
    return data


def load_test_data(data_path):
    # load the test csv file and return the data
    grass_data = pd.read_csv(data_path)
    data = []
    for tweet in grass_data.iterrows():
        data.append(preprocess(str(tweet[1].text)))
    return data


TRAIN_DATA_PATH = "train_data.csv"
EVAL_DATA_PATH = "eval_data.csv"
TEST_DATA_PATH = "test_data_nolabel.csv"
CLASSES = ['negative', 'neutral', 'positive']

def main():

    # train model and report the duration time
    start = time()
    nb_classifier = NaiveBayesClassifier(CLASSES)
    nb_classifier.train(load_data(TRAIN_DATA_PATH))
    end = time()
    print(f"training duration time:", round(end - start, 2), "seconds")


    # check on evaluation data
    eval_data = load_data(EVAL_DATA_PATH)
    all_tweets = 0
    correct_guess = 0
    for eval_tweet, label in eval_data:
        all_tweets += 1
        guess_label = nb_classifier.classify(eval_tweet)
        if guess_label == label:
            correct_guess += 1
    print(f"correctness percent on evaluation data: {round(correct_guess / all_tweets * 100, 2)}%")


    # test on test data:
    test_data = load_test_data(TEST_DATA_PATH)
    number = 0
    with open("labels.txt", "a") as f:
        for test_tweet in test_data:
            guess_label = nb_classifier.classify(test_tweet)
            output = str(number) + ": " + guess_label + "\n"
            f.write(output)
            number += 1



if __name__ == "__main__":
    main()

    # test_string = "I love playing football"
    # print(nb_classifier.classify(preprocess(test_string)))
