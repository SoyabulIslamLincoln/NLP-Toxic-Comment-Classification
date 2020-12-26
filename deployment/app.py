import numpy as np, pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template, request
import re

app = Flask(__name__)
model = keras.models.load_model('models/model.h5')

train = pd.read_csv('../dataset/train.csv')
list_sequences_train = train["comment_text"]
max_features = 22000
tokenizer = Tokenizer(num_words=max_features, oov_token='OOV')
train = tokenizer.fit_on_texts(list(list_sequences_train))

def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    comment = [x for x in request.form.values()]
    comment = [preprocess_text(comment[0])]
    
    tokenizer.fit_on_texts(comment)
    test = tokenizer.texts_to_sequences(comment)
    final_test = pad_sequences(test, padding='post', maxlen=200)

    prediction = model.predict(final_test)
    return render_template('index.html', prediction = prediction)

if __name__ == "__main__":
    app.run(debug=True)