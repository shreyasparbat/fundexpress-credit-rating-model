# Import libraries
import numpy as np
from flask import Flask, request, jsonify
from sklearn.externals import joblib

# Initiate flask
app = Flask(__name__)

# POST: test route
@app.route('/test', methods=['POST'])
def test():
    try:
        test_data = request.form['test_data']
        return jsonify({
            'input': test_data
        })
    except Exception as e:
        print(e)
    

# POST: predict default probability
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
    c_percent = predicted_probs[0, 0]
    d_percent = predicted_probs[0, 1]
    l_percent = predicted_probs[0, 2]

    # Set default credit rating
    credit_rating = 'B'
    ltv_percentage = 0.9

    # Get credit rating and ltv%
    if c_percent >= 0.7:
        credit_rating = 'A'
        ltv_percentage = 0.95
    elif c_percent >= 0.6:
        credit_rating = 'B'
        ltv_percentage = 0.90
    elif c_percent >= 0.2 and l_percent >= 0.2:
        credit_rating = 'C'
        ltv_percentage = 0.85
    elif d_percent <= 0.6:
        credit_rating = 'D'
        ltv_percentage = 0.80
    elif d_percent <= 0.7:
        credit_rating = 'E'
        ltv_percentage = 0.75
    elif d_percent <= 1:
        credit_rating = 'F'
        ltv_percentage = 0.70

    # Return predictions
    return jsonify({
        'cPercent': c_percent,
        'dPercent': d_percent,
        'lPercent': l_percent,
        'creditRating': credit_rating,
        'ltvPercentage': ltv_percentage
    })


# Run server
if __name__ == '__main__':
    clf = joblib.load('models/model_3.pkl')
    app.run(host='0.0.0.0')
