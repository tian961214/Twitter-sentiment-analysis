import numpy as np
import sys
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#from nltk.tokenize.moses import MosesDetokenizer
#  stopword
data_set_file=sys.argv[1]
test_set_file=sys.argv[2]
import pandas as pd
def predict_and_test(model, X_test_bag_of_words):
    predicted_y = model.predict(X_test_bag_of_words)
    output=pd.DataFrame()
    output[0] = test[0]
    output[1] = predicted_y
    output.to_csv('output.txt',header=None,sep='\t',index=None)
    f2 = open("output.txt", "r")
    lines = f2.readlines()
    for line3 in lines:
        print(line3,end="")

    #print(output)

import re
data=pd.read_csv(data_set_file,sep='\t',header=None)
test=pd.read_csv(test_set_file,sep='\t',header=None)

#print(data.info())
#data.columns=['index','tweet','attitude']
#test.columns=['index','tweet','attitude']
train_id=np.array(data[0])
train_sentence=np.array(data[1])
train_label=np.array(data[2])

test_id=np.array(test[0])
test_sentence=np.array(test[1])
test_label=np.array(test[2])

from nltk.stem.porter import *
# Delete twitter handle
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
       # input_txt = re.sub(r'@[A-Za-z0-9]+','',input_txt)
    return input_txt
# Remove punctuation, numbers and special characters
def process(data):

    data[1] = data[1].str.replace("[^a-zA-Z#]", " ")
    data[1] = np.vectorize(remove_pattern)(data[1], "@[\w]*")
# Delete short words
    data[1] = data[1].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
    #data[1] =data[1].str.lower()
    tokenized_tweet = data[1].apply(lambda x: x.split())
    #tokenized_tweet= tokenized_tweet.apply(lambda x: [item for item in x if item not in stop])

# Stemming

    stemmer = PorterStemmer()
    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
    #tokenized_tweet.head()
    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
        #tokenized_tweet[i] = detokenizer.detokenize(tokenized_tweet[i], return_str=True)

    data[1] = tokenized_tweet
    #data[1]= ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\S+)", " ", data[1]).split())
    return data
data=process(data)
test=process(test)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
bow_vectorizer = CountVectorizer(max_features=1000)
#bow_test_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(train_sentence)
bow_test=bow_vectorizer.transform(test_sentence)
x_train,y_train=bow,train_label
x_test,y_test=bow_test,test_label
#print(bow.shape,bow_test.shape)
#print("----DT")
lreg = LogisticRegression()
#clf = tree.DecisionTreeClassifier(min_samples_leaf=20,criterion='entropy',random_state=0)
model = lreg.fit(x_train, y_train)
predict_and_test(model, x_test)