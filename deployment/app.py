import numpy as np, pandas as pd
import tensorflow as tf 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.corpus import stopwords
set(stopwords.words('english'))

app = Flask(__name__)
model = tf.keras.models.load_model('models/model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    comment = [x for x in request.form.values()]
    comment = comment[0]  
    
    # Initiate a Tfidf vectorizer
    
    try:
        tfv = TfidfVectorizer(stop_words='english')
    except Exception as identifier:
        print(identifier)
    
    # Convert the X data into a document term matrix dataframe
    text = tfv.fit_transform(comment) 

    # max_features = 22000
    # tokenizer = Tokenizer(num_words=max_features)
    # tokenizer.fit_on_texts(comment)
    # tokenized_comment = tokenizer.texts_to_sequences(comment)
    # maxlen = 200
    # pad_comment = pad_sequences(tokenized_comment, maxlen = maxlen)

    #prediction = model.predict(text)
    return render_template('index.html', prediction = text)

if __name__ == "__main__":
    app.run(debug=True)