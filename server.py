# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import joblib
app = Flask(__name__)
# Load the model
model= joblib.load('reg_1.pkl')

@app.route('/api',methods=['POST'])
def predict():

    data = request.get_json(force=True)
    data = [[data["age"], data["gender"], data["height"], data["weight"], data["smoke"], data["alco"], data["active"]]]

    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict(data)
    # Take the first value of prediction
    output = prediction[0]

    return {"result": int(output)}

if __name__ == '__main__':
    app.run(port=5000, debug=True,use_reloader=False)