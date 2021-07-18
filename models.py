from matplotlib import pyplot as plt
from pytesseract import Output
import pytesseract
import cv2
import os
import re
from collections import namedtuple
from PIL import Image
import sys
from pdf2image import convert_from_path
import pyocr
import pyocr.builders
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import pickle

import string
import re

import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, classification_report

stop = stopwords.words('english')
porter = PorterStemmer()

def remove_punctuation(Text):
    """The function to remove punctuation"""
    table = str.maketrans('', '', string.punctuation)
    return Text.translate(table)

def remove_stopwords(text):
    """The function to removing stopwords"""
    text = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(text)

def stemmer(stem_text):
    """The function to apply stemming"""
    stem_text = [porter.stem(word) for word in stem_text.split()]
    return " ".join(stem_text)
    
print('loading the data')
# Load dataset
url = "proof-of-payments labels.xlsx - Sheet1.csv"
f = pd.read_csv(url)
print('Cleaning the data')
#clean the dataset
f.replace('Internet', 'Internet Payment', inplace=True)
f.rename(columns= {'Proof_of_Payments Refs': 'Proof_of_Payments_Refs'}, inplace=True)
f.loc[f.Proof_of_Payments_Refs == '1608369646_467409.', "Source"] = "Sms Screenshot"
f.loc[f.Proof_of_Payments_Refs == '1608369696_636015.', "Source"] = "Sms Screenshot"
f.loc[f.Proof_of_Payments_Refs == '1608370632_612430.', "Source"] = "Sms Screenshot"
f.loc[f.Proof_of_Payments_Refs == '1608370681_486993.', "Source"] = "Sms Screenshot"
f.loc[f.Proof_of_Payments_Refs == '1611840255_817553.', "Source"] = "Browser Screenshot"
f.loc[f.Proof_of_Payments_Refs == '1611840308_917195.', "Source"] = "Browser Screenshot"
f.loc[f.Proof_of_Payments_Refs == '1613030160_248235.pdf', "Source"] = "Rental Invoice"
f.loc[f.Proof_of_Payments_Refs == '1613030303_184939.pdf', "Source"] = "Rental Invoice"
f.loc[f.Proof_of_Payments_Refs == '1613030491_481261.pdf', "Source"] = "Rental Invoice"
f.loc[f.Proof_of_Payments_Refs == '1613030653_380748.pdf', "Source"] = "Rental Invoice"
f.loc[f.Proof_of_Payments_Refs == '1613030995_66724.pdf', "Source"] = "Rental Invoice"
f.loc[f.Proof_of_Payments_Refs == '1613031112_710473.pdf', "Source"] = "Rental Invoice"
f.loc[f.Proof_of_Payments_Refs == '1613126378_164420.pdf', "Source"] = "Rental Invoice"
f.loc[f.Proof_of_Payments_Refs == '1613129088_954254.pdf', "Source"] = "Rental Invoice"
f.loc[f.Proof_of_Payments_Refs == '1613129136_726526.pdf', "Source"] = "Rental Invoice"
f.loc[f.Proof_of_Payments_Refs == '1613129164_135887.pdf', "Source"] = "Rental Invoice"
f.loc[f.Proof_of_Payments_Refs == '1613129206_618154.pdf', "Source"] = "Rental Invoice"
f.loc[f.Proof_of_Payments_Refs == '1613129340_203858.pdf', "Source"] = "Rental Invoice"
f.loc[f.Proof_of_Payments_Refs == '1613129384_603736.pdf', "Source"] = "Rental Invoice"
f.loc[f.Proof_of_Payments_Refs == '1613129484_769632.pdf', "Source"] = "Rental Invoice"
f.loc[f.Proof_of_Payments_Refs == '1613129536_547208.pdf', "Source"] = "Rental Invoice"
f.loc[f.Proof_of_Payments_Refs == '1613129573_985797.pdf', "Source"] = "Rental Invoice"
f.loc[f.Proof_of_Payments_Refs == '1613129615_800940.pdf', "Source"] = "Rental Invoice"
f.loc[f.Proof_of_Payments_Refs == '1613129653_957093.pdf', "Source"] = "Rental Invoice"
f.loc[f.Proof_of_Payments_Refs == '1613129684_80997.pdf', "Source"] = "Rental Invoice"
f.loc[f.Proof_of_Payments_Refs == '1613129713_239655.pdf', "Source"] = "Rental Invoice"
f.loc[f.Proof_of_Payments_Refs == '1613129749_816835.pdf', "Source"] = "Rental Invoice"
f.loc[f.Proof_of_Payments_Refs == '1614690912_255735.pdf', "Source"] = "Rental Invoice"
f.loc[f.Proof_of_Payments_Refs == '1616659735_735563.', "Source"] = "Sms Screenshot"
f.loc[f.Proof_of_Payments_Refs == '1617259387_99073.', "Source"] = "Sms Screenshot"
f.loc[f.Proof_of_Payments_Refs == '1617259583_942240.', "Source"] = "Sms Screenshot"
f.loc[f.Proof_of_Payments_Refs == '1617259640_70783.', "Source"] = "Sms Screenshot"
f.loc[f.Proof_of_Payments_Refs == '1617259704_655699.', "Source"] = "Sms Screenshot"
f.loc[f.Proof_of_Payments_Refs == '1618554477_615854.pdf', "Source"] = "Rental Invoice"
f.loc[f.Proof_of_Payments_Refs == '1618554587_213274.pdf', "Source"] = "Rental Invoice"
f.loc[f.Proof_of_Payments_Refs == '1618554661_176570.pdf', "Source"] = "Rental Invoice"
f.loc[f.Proof_of_Payments_Refs == '1618554749_718840.pdf', "Source"] = "Rental Invoice"
f.loc[f.Proof_of_Payments_Refs == '1618554817_197078.pdf', "Source"] = "Rental Invoice"
f.loc[f.Proof_of_Payments_Refs == '1618908346_244066.pdf', "Source"] = "Rental Invoice"
f.loc[f.Proof_of_Payments_Refs == '1618908814_9311.pdf', "Source"] = "Rental Invoice"
f.loc[f.Proof_of_Payments_Refs == '1618908906_95625.pdf', "Source"] = "Rental Invoice"
f.loc[f.Proof_of_Payments_Refs == '1618908980_880090.pdf', "Source"] = "Rental Invoice"
f.loc[f.Proof_of_Payments_Refs == '1619710412_935460.pdf', "Source"] = "Rental Invoice"
f.drop('details',axis=1,inplace=True)

f['Text'] = f['Text'].astype(str)
f2=f.drop_duplicates()
f2.dropna(subset=['Source'], inplace=True)
f2.dropna(subset=['Bank'], inplace=True)

f2['Text'] = f2['Text'].apply(remove_punctuation)
f2['Text'] = f2['Text'].apply(remove_stopwords)
f2['Text'] = f2['Text'].apply(stemmer)

data = f2.to_csv('clean.csv', encoding='utf-8')

vectorizer = CountVectorizer()
vectorizer.fit(f2['Text'])
vector = vectorizer.transform(f2['Text'])

tfidf_converter = TfidfTransformer()
X_tfidf = tfidf_converter.fit_transform(vector).toarray()

X = f2['Text']
y = f2['Source']
print('Building models....')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)
#SVM model
svc = Pipeline([('vect', CountVectorizer(min_df=5, ngram_range=(1,2))),
               ('tfidf', TfidfTransformer()),
               ('model',LinearSVC()),
               ])

svc.fit(X_train, y_train)

ytest = np.array(y_test)
y_pred = svc.predict(X_test)


print(classification_report(ytest, y_pred))

# Logistic regression model
model_log = Pipeline([('vect', CountVectorizer(min_df=5, ngram_range=(1,2))),
                      ('tfidf', TfidfTransformer()),
                      ('model',LogisticRegression()),
                     ])

model_log.fit(X_train, y_train)

ytest = np.array(y_test)
pred = model_log.predict(X_test)
print('Evaluating models')

print(classification_report(ytest, pred))

print('Saving the models')
# Save the model
with open("text_model.pkl", "wb") as model_file:
    pickle.dump(svc, model_file)

# Save the model
with open("text_model2.pkl", "wb") as model_file:
    pickle.dump(model_log, model_file)






    

    