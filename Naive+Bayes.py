
# coding: utf-8

# In[4]:


# Naive Bayes using NLP

# USe following code if it wont work in first place with UTF-8 code error

# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
import os

os.chdir("C:\\Users\\manishk.bajpai\\Desktop\\")

import csv

smsdata = open('SMSSpamCollection.txt','r')
csv_reader = csv.reader(smsdata,delimiter='\t')

smsdata_data = []
smsdata_labels = []

for line in csv_reader:
    smsdata_labels.append(line[0])
    smsdata_data.append(line[1])

smsdata.close()

# Printing top 5 lines
for i in range(5):
    print (smsdata_data[i],smsdata_labels[i])


# In[5]:


# Printing Spam & Ham count
from collections import Counter
c = Counter( smsdata_labels )
print(c)


# In[9]:


import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import pandas as pd
from nltk import pos_tag
from nltk.stem import PorterStemmer
import nltk
nltk.download('popular')

def preprocessing(text):
    text2 = " ".join("".join([" " if ch in string.punctuation else ch for ch in text]).split())

    tokens = [word for sent in nltk.sent_tokenize(text2) for word in
              nltk.word_tokenize(sent)]
    
    tokens = [word.lower() for word in tokens]
    
    stopwds = stopwords.words('english')
    tokens = [token for token in tokens if token not in stopwds]
    
    tokens = [word for word in tokens if len(word)>=3]
    
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    tagged_corpus = pos_tag(tokens)    
    
    Noun_tags = ['NN','NNP','NNPS','NNS']
    Verb_tags = ['VB','VBD','VBG','VBN','VBP','VBZ']

    lemmatizer = WordNetLemmatizer()

    def prat_lemmatize(token,tag):
        if tag in Noun_tags:
            return lemmatizer.lemmatize(token,'n')
        elif tag in Verb_tags:
            return lemmatizer.lemmatize(token,'v')
        else:
            return lemmatizer.lemmatize(token,'n')
    
    pre_proc_text =  " ".join([prat_lemmatize(token,tag) for token,tag in tagged_corpus])             

    return pre_proc_text


smsdata_data_2 = []

for i in smsdata_data:
    smsdata_data_2.append(preprocessing(i))


import numpy as np


trainset_size = int(round(len(smsdata_data_2)*0.70))


print ('The training set size for this classifier is ' + str(trainset_size) + '\n')



# In[10]:


x_train = np.array([''.join(rec) for rec in smsdata_data_2[0:trainset_size]])
y_train = np.array([rec for rec in smsdata_labels[0:trainset_size]])
x_test = np.array([''.join(rec) for rec in smsdata_data_2[trainset_size+1:len(smsdata_data_2)]])
y_test = np.array([rec for rec in smsdata_labels[trainset_size+1:len(smsdata_labels)]])


# building TFIDF vectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2),  stop_words='english', 
                             max_features= 4000,strip_accents='unicode',  norm='l2')

x_train_2 = vectorizer.fit_transform(x_train).todense()
x_test_2 = vectorizer.transform(x_test).todense()

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(x_train_2, y_train)

ytrain_nb_predicted = clf.predict(x_train_2)
ytest_nb_predicted = clf.predict(x_test_2)

from sklearn.metrics import classification_report,accuracy_score

print ("\nNaive Bayes - Train Confusion Matrix\n\n",pd.crosstab(y_train,ytrain_nb_predicted,rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nNaive Bayes- Train accuracy",round(accuracy_score(y_train,ytrain_nb_predicted),3))
print ("\nNaive Bayes  - Train Classification Report\n",classification_report(y_train,ytrain_nb_predicted))

print ("\nNaive Bayes - Test Confusion Matrix\n\n",pd.crosstab(y_test,ytest_nb_predicted,rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nNaive Bayes- Test accuracy",round(accuracy_score(y_test,ytest_nb_predicted),3))
print ("\nNaive Bayes  - Test Classification Report\n",classification_report(y_test,ytest_nb_predicted))


# In[11]:


# printing top features 
feature_names = vectorizer.get_feature_names()
coefs = clf.coef_
intercept = clf.intercept_
coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))

print ("\n\nTop 10 features - both first & last\n")
n=10
top_n_coefs = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
for (coef_1, fn_1), (coef_2, fn_2) in top_n_coefs:
    print('\t%.4f\t%-15s\t\t%.4f\t%-15s' % (coef_1, fn_1, coef_2, fn_2))

