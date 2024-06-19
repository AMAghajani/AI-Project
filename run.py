from template import NaiveBayesClassifier
import pandas as pd
import re
import unidecode
from nltk.stem import PorterStemmer
from time import time

def preprocess(tweet_string):
    # clean the data and tokenize it
    str = re.sub(r"[&!\"#$%&()*+-./:;<=>?@[\]^_{|}~\n -' 0123456789\\]"," ", tweet_string)
    str = unidecode.unidecode(str)
    strs = str.split(' ')
    features = []
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
    # load the train csv file and return the data
    grass_data = pd.read_csv(data_path)
    data = []
    for tweet in grass_data.iterrows():
        data.append((preprocess(str(tweet[1].text)), str(tweet[1].label_text)))
    return data


def load_test_data(data_path):

    grass_data = pd.read_csv(data_path)
    data = []
    for tweet in grass_data.iterrows():
        data.append(preprocess(str(tweet[1].text)))
    return data



# train your model and report the duration time
start = time()
train_data_path = "train_data.csv"
eval_data_path = "eval_data.csv"
test_data_path = "test_data_nolabel.csv"
classes = ['negative', 'neutral', 'positive']
nb_classifier = NaiveBayesClassifier(classes)
nb_classifier.train(load_data(train_data_path))
end = time()
print(f"training duration time:", round(end - start, 2), "second")


# check on evaluation data
eval_data = load_data(eval_data_path)
all_tweets = 0
correct_guess = 0
for eval_tweet, label in eval_data:
    all_tweets += 1
    guess_label = nb_classifier.classify(eval_tweet)
    if guess_label == label:
        correct_guess += 1
print(f"correctness percent on evaluation data: {round(correct_guess / all_tweets * 100, 2)}%")


# test on test data:
test_data = load_test_data(test_data_path)
number = 0
f = open("labels.txt", "a")
for test_tweet in test_data:
    guess_label = nb_classifier.classify(test_tweet)
    output = str(number) + ": " + guess_label + "\n"
    f.write(output)
    number += 1
f.close()

# test_string = "I love playing football"
# print(nb_classifier.classify(preprocess(test_string)))
