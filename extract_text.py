#%% Imports

# Library imports
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import cv2

def get_charaters(image, height_resised=512, width_resised=512, confidence_limit=0.5, padding=5):
	## Image preprocessing

	# Resise image
	image = cv2.resize(image, (height_resised, width_resised))

	# Display image
	cv2.imshow("Image", image)
	cv2.waitKey(0)

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

	# # Applying adaptive thresholding
	# image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
	# cv2.imshow('Adaptive threshold', image)
	# cv2.waitKey(0)

	# # Applying gaussian blur
	# image = cv2.GaussianBlur(image, (7, 7), 0)
	# cv2.imshow('Gaussian blur', image)
	# cv2.waitKey(0)

	# Loop over the bounding boxes
	results = []
	for (startX, startY, endX, endY) in boxes:
		# Apply padding to bounding boxes
		startX = max(0, startX - padding)
		startY = max(0, startY - padding)
		endX = min(width_resised, endX + padding)
		endY = min(height_resised, endY + padding)
	
		# Draw the bounding box on the image (TESING PURPOSES ONLY)
		cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

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
	results = sorted(results, key=lambda r:r[0][1])
	
	# Loop over the results
	ocr_outputs = []
	for ((startX, startY, endX, endY), text) in results:
		# display the text OCR'd by Tesseract
		print("OCR TEXT")
		print("========")
		print("{}\n".format(text))
	
		# Strip out non-ASCII text so we can draw the text on the image
		# using OpenCV, then draw the text and a bounding box surrounding
		# the text region of the input image
		text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
		cv2.putText(image, text, (startX, startY - 20),
			cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
		)
		
		# Add outputs to list to be returned
		ocr_outputs.append(text)
	

	# Show the output image
	cv2.imshow("Text detected", image)
	cv2.waitKey(0)

	# Return all ocr outputs
	return ocr_outputs

print(get_charaters(cv2.imread('./images/bar.jpg')))


# image = cv2.imread('./images/pamp4.jpg')
# config = ("-l eng --oem 1 --psm 7")
# text = pytesseract.image_to_string(image, config=config)
# print(text)	
