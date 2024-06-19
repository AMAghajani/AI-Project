# Naive Bayes 3-class Classifier 
# Authors: Baktash Ansari - Sina Zamani 

# complete each of the class methods  

from math import log10

class NaiveBayesClassifier:

    def __init__(self, classes):
        # initialization: 
        # inputs: classes(list) --> list of label names
        # class_word_counts --> frequency dictionary for each class
        # class_counts --> number of instances of each class
        # vocab --> all unique words  
        self.classes = classes
        self.class_word_counts = {'negative': 0, 'neutral': 0, 'positive': 0}
        self.class_counts = {'negative': 0, 'neutral': 0, 'positive': 0}
        self.vocab = {}
        self.tweet_counts = None

    def train(self, data):
        # training process:
        # inputs: data(list) --> each item of list is a tuple 
        # the first index of the tuple is a list of words and the second index is the label(positive, negative, or neutral)
        self.tweet_counts = len(data)
        for features, label in data:
            self.class_counts[label] += 1
            for feature in features:
                self.class_word_counts[label] += 1
                if (feature, label) in self.vocab.keys():
                    self.vocab[(feature, label)] += 1
                else:
                    self.vocab[(feature, label)] = 1

    def calculate_prior(self, label):
        # calculate log prior
        # you can add some attributes to this method
  
        return log10(self.class_counts[label] / self.tweet_counts)

    def calculate_likelihood(self, word, label):
        # calculate likelihhood: P(word | label)
        # return the corresponding value

        if (word, label) in self.vocab.keys():
            return log10((self.vocab[(word, label)] + 1) / (self.class_word_counts[label] + 3))
        return log10(1 / (self.class_word_counts[label] + 3))

    def classify(self, features):
        # predict the class
        # inputs: features(list) --> words of a tweet 
        
        what_class = {'negative': 0, 'neutral': 0, 'positive': 0}

        for label in self.classes:
            for feature in features:
                what_class[label] += self.calculate_likelihood(feature, label)
            what_class[label] += self.calculate_prior(label)

        best_class = max(what_class, key=what_class.get)

        return best_class
    
# Good luck :)
