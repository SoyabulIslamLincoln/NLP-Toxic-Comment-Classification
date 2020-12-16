import numpy as np, pandas as pd
import tensorflow as tf 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template, request

app = Flask(__name__)
model = tf.keras.models.load_model('models/model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    comment = [x for x in request.form.values()]
    comment = comment[0]
    
    #tokenizer = Tokenizer(num_words=20000)
    #tokenized_comment = tokenizer.texts_to_sequences(comment)
    #padded_comment = pad_sequences(tokenized_comment, maxlen=10)
    
    #prediction = model.predict(padded_comment)
    
    return render_template('index.html', comment = comment)

if __name__ == "__main__":
    app.run(debug=True)