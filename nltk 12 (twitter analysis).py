from pyexpat import native_encoding
import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize

from statistics import mode

#####

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    def confidence(self,features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes)) #counts how many occurences of the most frequent vote there are
        conf = choice_votes / len(votes) # #of frequent votes/#of total votes
        return conf


short_pos = open('E:/positive.txt','r').read()
short_neg = open('E:/negative.txt','r').read()

all_words = []
documents = []

#  j is adject, r is adverb, and v is verb
#allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]

for r in short_pos.split('\n'):  #we're splitting the reviews by line
    documents.append( (r,'pos')) #we're adding the positive reviews to the list\
    words = word_tokenize(r) #we're separating the words in each pos review
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower()) 

for r in short_neg.split('\n'):
    documents.append( (r,'neg'))  #we're adding the negative reviews to the list
    words = word_tokenize(r)
    pos = nltk.pos_tag(words)  #we're separating the words in each neg review
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

save_documents = open('documents.pickle','wb')
pickle.dump(documents, save_documents)
save_documents.close()

all_words = nltk.FreqDist(all_words) #FredDist assigns each word avalue of how many times it appears

# print(all_words.most_common(15))  #mostcommon chooses the top 15 highest frequency words
#all_words includes punctuation and filler words, so not that useful

word_features = list(all_words.keys())[:3000]  #all_words.keys includes just the words and not the count of how many times they appear

save_word_features = open('word_features5k.pickle','wb')
pickle.dump(word_features, save_word_features)
save_word_features.close()
    
def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)  #saying if one of the top 3000 words in word_features is in the set of words, return true, else false

    return features

#print(find_features(movie_reviews.words('neg/cv000_29416.txt'))) #this searchses if this movie review has any top 3000 words

featuresets = [(find_features(rev), category) for (rev,category) in documents] 

random.shuffle(featuresets)
print(len(featuresets))

#positive data
training_set = featuresets[:10000]
testing_set = featuresets[10000:]

#negative data
# training_set = featuresets[100:]
# testing_set = featuresets[:100]

#posterior (means likelihood)  = (prior ocurrences X likelihood) / evidence

#classifier = nltk.NaiveBayesClassifier.train(training_set) uses the algo on the training data

# classifier_f = open('naivebayes.pickle', 'rb') #saves the algo as a file
# classifier = pickle.load(classifier_f) #opens the file
# classifier_f.close()

#####

classifier = nltk.NaiveBayesClassifier.train(training_set)
print('Naive Bayes Algo accuracy percent', (nltk.classify.accuracy(classifier, testing_set))*100) #prints accuracy percent of algorithm
classifier.show_most_informative_features(15) 

save_classifier = open('originalnaivebayes5k.pickle','wb') #saves the classifier as a file
pickle.dump(classifier, save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print('MNB_classifier accuracy percent', (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

save_classifier = open('MNB_classifier5k.pickle','wb') #saves the classifier as a file
pickle.dump(classifier, save_classifier)
save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print('BernoulliNB_classifier accuracy percent', (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

save_classifier = open('BenoulliNB_classifier5k.pickle','wb') #saves the classifier as a file
pickle.dump(classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print('LogisticRegression_classifier accuracy percent', (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

save_classifier = open('LogisticRegression_classifier5k.pickle','wb') #saves the classifier as a file
pickle.dump(classifier, save_classifier)
save_classifier.close()

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print('SGDClassifier_classifier accuracy percent', (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

save_classifier = open('SDGC_classifier5k.pickle','wb') #saves the classifier as a file
pickle.dump(classifier, save_classifier)
save_classifier.close()

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print('LinearSVC_classifier accuracy percent', (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

save_classifier = open('LinearSVC_classifier5k.pickle','wb') #saves the classifier as a file
pickle.dump(classifier, save_classifier)
save_classifier.close()

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print('NuSVC_classifier accuracy percent', (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

save_classifier = open('NuSVC_classifier5k.pickle','wb') #saves the classifier as a file
pickle.dump(classifier, save_classifier)
save_classifier.close()

# Combining Algos with a vote

voted_classifier = VoteClassifier(classifier, 
                                MNB_classifier, 
                                BernoulliNB_classifier, 
                                LogisticRegression_classifier, 
                                SGDClassifier_classifier, 
                                LinearSVC_classifier, 
                                NuSVC_classifier)

print('voted_classifier accuracy percent', (nltk.classify.accuracy(voted_classifier, testing_set))*100)

def sentiment(text):
    feats = find_features(text)
    return feats
