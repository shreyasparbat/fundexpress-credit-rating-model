# Import libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
from imutils.object_detection import non_max_suppression
import pytesseract
import cv2
import requests
import time
import imageio

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
    except:
        return traceback.format_exc()


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
    else:
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
    else:
        features.append(0)
        features.append(0)
        features.append(1)

    # Add Phone
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
    clf = joblib.load('models/model.pkl')
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


# POST: Retrain credit rating model
@app.route('/retrainCreditRatingModel', methods=['POST'])
def retrain():
    try:
        # Get and parse file
        file = request.form['fileUpload']
        data = pd.read_excel(file)
        print(data.head())

        # Get dummies and save dependent variable
        status_list = data.Status.tolist()
        model_3_df = model_3_df.drop(['Status'], axis=1)
        model_3_dummied_df = pd.get_dummies(model_3_df)
        model_3_dummied_df['Status'] = status_list
        model_3_dummied_df.head()

        # Get dependent and independent variable arrays
        x = model_3_dummied_df.iloc[:, :-1].values
        y = model_3_dummied_df.iloc[:, -1].values

        # Get training and test sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

        # Create and fit classifier
        classifier = RandomForestClassifier(n_estimators=300, criterion='entropy', random_state=0)
        classifier.fit(x_train, y_train)

        # Pickle and save model
        joblib.dump(classifier, 'models/model.pkl')
    except:
        return traceback.format_exc()


# POST: predict default probability
@app.route('/bar_ocr', methods=['POST'])
def get_characters():
    # Get images
    item_id = request.form['itemID']
    front_url = 'https://fundexpress-api-storage.sgp1.digitaloceanspaces.com/item-images/' + str(item_id) + '_front.jpg'
    back_url = 'https://fundexpress-api-storage.sgp1.digitaloceanspaces.com/item-images/' + str(item_id) + '_back.jpg'
    print(back_url)
    time.sleep(5)
    front_image = requests.get(front_url)
    back_image = requests.get(back_url)

    # Run OCR on both images
    front_output = run_ocr(imageio.imread(front_image.content))
    back_output = run_ocr(imageio.imread(back_image.content))

    # Return json response
    return jsonify({
        'front_text': front_output,
        'back_text': back_output
    })


# HELPER: to process the image and run some ocr on it
def run_ocr(image, resise_factor=20, confidence_limit=0.5, padding=7):
    ## Image preprocessing

    # Resise image
    height_resised = 32 * resise_factor
    width_resised = 32 * resise_factor
    image = cv2.resize(image, (height_resised, width_resised))

    # Save orginial image for drawing puposes later
    originial_image = image

    # Display image
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    # cv2.imwrite('images/original.jpg', image)

    ## Text detection using EAST

    # Load pre-trained text detector
    net = cv2.dnn.readNet('frozen_east_text_detection.pb')

    # Define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    # Construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (height_resised, width_resised),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    scores, geometry = net.forward(layerNames)

    # Grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # Loop over the number of rows
    for y in range(0, numRows):
        # Extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # Loop over the number of columns
        for x in range(0, numCols):
            # If our score does not have sufficient probability, ignore it
            if scoresData[x] < confidence_limit:
                continue

            # Compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # Extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # Use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # Compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # Add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # Apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    ## Running OCR

    # Gray scale image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applying adaptive thresholding
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    # cv2.imshow('Adaptive threshold', image)
    # cv2.waitKey(0)
    # cv2.imwrite('images/thresholded.jpg', image)

    # Applying gaussian blur
    # image = cv2.GaussianBlur(image, (7, 7), 0)
    # cv2.imshow('Gaussian blur', image)
    # cv2.waitKey(0)

    # Loop over the bounding boxes and get text
    results = []
    for (startX, startY, endX, endY) in boxes:
        # Apply padding to bounding boxes
        startX = max(0, startX - padding)
        startY = max(0, startY - padding)
        endX = min(width_resised, endX + padding)
        endY = min(height_resised, endY + padding)

        # Extract bounding box as roi (region of interest)
        roi = image[startY:endY, startX:endX]
        roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

        # In order to apply Tesseract v4 to OCR text we must supply
        # (1) a language, (2) an OEM flag of 4, indicating that the we
        # wish to use the LSTM neural net model for OCR, and finally
        # (3) an OEM value, in this case, 7 which implies that we are
        # treating the ROI as a single line of text
        config = ("-l eng --oem 1 --psm 7")
        text = pytesseract.image_to_string(roi, config=config)

        # add the bounding box coordinates and OCR'd text to the list
        # of results
        results.append(((startX, startY, endX, endY), text))

    ## Printing result

    # Sort the results bounding box coordinates from top to bottom
    results = sorted(results, key=lambda r: r[0][1])

    # Loop over the results
    ocr_outputs = []
    marked_image = originial_image
    for ((startX, startY, endX, endY), text) in results:
        # display the text OCR'd by Tesseract
        # print("OCR TEXT")
        # print("========")
        # print("{}\n".format(text))

        # Draw the bounding box on the image (TESING PURPOSES ONLY)
        cv2.rectangle(marked_image, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Strip out non-ASCII text so we can draw the text on the image
        # using OpenCV, then draw the text and a bounding box surrounding
        # the text region of the input image
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        cv2.putText(marked_image, text, (startX, startY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
                    )

        # Add outputs to list to be returned
        ocr_outputs.append(text)

    # Show the output image
    # cv2.imshow("Text detected", marked_image)
    # cv2.waitKey(0)
    # cv2.imwrite('images/final.jpg', marked_image)

    # Return all ocr outputs
    return ocr_outputs


# Run server
if __name__ == '__main__':
    app.run(host='0.0.0.0')
