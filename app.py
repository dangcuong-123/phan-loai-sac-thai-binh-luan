# Dependencies
from flask import Flask, request, jsonify
import flask
import joblib
import traceback
import pandas as pd
import numpy as np
import re 
from pyvi import ViTokenizer, ViPosTagger

# Your API definition
app = Flask(__name__)

def preprocessor(text):
    """ Return a cleaned version of text
    """
    # Remove HTML markup
    text = re.sub('<[^>]*>', '', text)
    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    text = re.sub(r"[\.,\?]+$-", "", text)
    # Save emoticons for later appending
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    
    # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
    text = text.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ") \
        .replace("-", " ").replace("?", " ")
 
    text = text.strip()
    # Remove any non-word character and append the emoticons,
    # removing the nose character for standarization. Convert to lower case
    text = (re.sub('[\W]+', ' ', text.lower()) + ' ' + ' '.join(emoticons).replace('-', ''))
    
    return text

def tokenizer(text):
    return ViTokenizer.tokenize(text)

@app.route('/predict', methods=['GET'])
def predict():
    if model:
        try:
            list_prediction = []
            json_ = flask.request.json
            for i in range(len(json_)):
                print(json_[i]['message'])
                prediction = model.predict_proba([json_[i]['message']])
                list_prediction.append(prediction[0][0])
            return jsonify({'prediction': str(list_prediction)})

        except:
        	return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')


if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    model = joblib.load('model_final.joblib') # Load model
    print ('Model loaded')

    app.run(port=port, debug=True)