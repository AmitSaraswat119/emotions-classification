from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
import spacy

nlp = spacy.load("en_core_web_lg")
spacy.prefer_gpu()
app = Flask(__name__)
model = load_model('emotion_classification.h5')
emotions_map = {0: "sadness", 1: "joy", 2: "neutral", 3: "anger", 4: "fear"}

def prediction(text):
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)
    prediction_string = " ".join(filtered_tokens)
    doc = nlp(prediction_string)
    embeddings = doc.vector
    embeddings = np.stack([embeddings])
    prediction = model.predict(embeddings)
    prediction = np.argmax(prediction[0])
    for key, val in emotions_map.items():
        if key == prediction:
            return val
    

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify_emotion', methods=['POST'])
def classify_emotion():
    text = request.form['textInput']
    emotion_label = prediction(text)
    return render_template('index.html', emotion=emotion_label)

if __name__ == '__main__':
    webbrowser.open('http://127.0.0.1:5000')
    app.run(debug=True , port=5000)
