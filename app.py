# Import libraries
import numpy as np
from flask import Flask, request, jsonify
from sklearn.externals import joblib

# Initiate flask
app = Flask(__name__)


# GET: predict default probability
@app.route('/predict', methods=['POST'])
def make_prediction():
    # Get data
    age = request.form['age']
    nric = request.form['nric']
    race = request.form['race']
    sex = request.form['sex']
    nation = request.form['nation']
    address = request.form['address']
    tel = request.form['tel']

    # Create feature list and add age
    features = [age]

    # Add nric
    if nric == 'F':
        features.append(1)
        features.append(0)
    else:
        features.append(0)
        features.append(1)

    # Add race
    if race == 'C':
        features.append(1)
        features.append(0)
        features.append(0)
        features.append(0)
    elif race == 'I':
        features.append(0)
        features.append(1)
        features.append(0)
        features.append(0)
    elif race == 'M':
        features.append(0)
        features.append(0)
        features.append(1)
        features.append(0)
    else:
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(1)

    # Add sex
    if sex == 'F':
        features.append(1)
        features.append(0)
    else:
        features.append(0)
        features.append(1)

    # Add nation
    if nation == 'F':
        features.append(1)
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(0)
    elif nation == 'I':
        features.append(0)
        features.append(1)
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(0)
    elif nation == 'M':
        features.append(0)
        features.append(0)
        features.append(1)
        features.append(0)
        features.append(0)
        features.append(0)
    elif nation == 'P':
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(1)
        features.append(0)
        features.append(0)
    elif nation == 'S':
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(1)
        features.append(0)
    elif nation == 'T':
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(1)

    # Add address
    if address == 'C':
        features.append(1)
        features.append(0)
        features.append(0)
    elif address == 'H':
        features.append(0)
        features.append(1)
        features.append(0)
    elif address == 'N':
        features.append(0)
        features.append(0)
        features.append(1)

    if tel == 'H':
        features.append(1)
        features.append(0)
    else:
        features.append(0)
        features.append(1)

    # Convert to numpy array
    features_array = np.asarray(features).reshape(1, -1)
    print(features_array)
    print(type(features_array))
    print(features_array.shape)

    # Make predictions
    predicted_probs = clf.predict_proba(features_array)

    # Return predictions
    return jsonify({
        'C': predicted_probs[0, 0],
        'D': predicted_probs[0, 1],
        'L': predicted_probs[0, 2]
    })


# Run server
if __name__ == '__main__':
    clf = joblib.load('models/model_3.pkl')
    app.run(host='0.0.0.0')
