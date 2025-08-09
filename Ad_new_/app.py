from flask import Flask, render_template, request
import pickle
import numpy as np
from feature.feature import FeatureExtraction 

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('pickle/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        url = request.form['url']  # To Get the input URL
        
        # Extract features using the FeatureExtraction class
        extractor = FeatureExtraction(url)
        url_features = np.array(extractor.getFeaturesList()).reshape(1, -1)
        
        # Make a prediction using the model
        prediction = model.predict(url_features)[0]

        if prediction == 1:
            result = "This is a safe website."
        elif prediction == 0:
            result = "This website is suspicious."
        else:
            result = "This is a phishing website."

        return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
