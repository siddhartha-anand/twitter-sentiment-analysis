import streamlit as st
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle as pk
import joblib
import nltk

st.title('Twitter Sentiment Analysis')

user_input = st.text_input('Paste your tweet here...')

if st.button('Analyse'):
  nltk.download('stopwords')

  port_stem = PorterStemmer()
  stemmed_content = re.sub('[^a-zA-Z]',' ',user_input)#remove everything that isn't a letter
  stemmed_content = stemmed_content.lower()
  stemmed_content = stemmed_content.split()
  stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content = ' '.join(stemmed_content)

  test = []
  test.append(stemmed_content)

  vectorizer = joblib.load('vectorizer.pkl')
  test = vectorizer.transform(test)

  model = pk.load(open('trained_model.sav','rb'))

  prediction = model.predict(test)
  print(prediction)

  if (prediction[0]==0):
    st.write('Negative Tweet')
  else:
    st.write('Positive Tweet')