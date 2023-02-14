from flask import Flask, jsonify, request
from tensorflow import keras
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # get the news title from the request
    X = request.json['title']

    # load the dataset
    model = keras.models.load_model('fakenewsmodel.h5')
    dataraw = pd.read_csv("news.csv")
    data = dataraw.drop(["Unnamed: 0"], axis=1)

    # encode the labels
    le = preprocessing.LabelEncoder()
    le.fit(data['label'])
    data['label'] = le.transform(data['label'])

    # fit the tokenizer on the news titles
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data['title'])

    # tokenize the news title
    sequences = tokenizer.texts_to_sequences(X)[0]
    sequences = pad_sequences([sequences], maxlen=54,
                              padding='post',
                              truncating='post')

    # make a prediction
    prediction = model.predict(sequences, verbose=0)[0][0]

    # return the prediction as a JSON response
    if prediction >= 0.8:
        result = {"prediction": "This news is probably true"}
    elif prediction >= 0.5:
        result = {"prediction": "This news could be true"}
    elif prediction <= 0.2:
        result = {"prediction": "This news is probably fake"}
    else:
        result = {"prediction": "This news could be fake"}

    return jsonify(result)

if __name__ == '__main__':
    app.run()
