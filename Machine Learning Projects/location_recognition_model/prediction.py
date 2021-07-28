# import np and tf and set seeds before anything else
import keras
import numpy as np
import tensorflow as tf

import json
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import os
from time import perf_counter
import cv2.cv2 as cv2
from sklearn.model_selection import train_test_split
from datetime import datetime
from PIL import Image, ImageFilter

def Reader_prediction(path, size_x=1700, size_y=2338):
        img = plt.imread(path)
        img = np.array(img[:, :, 0:3])

        dataFrame['image'].append(cv2.resize(img, (size_x, size_y)))


def Iterator(imageDir):
    imageNames = os.listdir(imageDir)

    for mem in imageNames:
        path = imageDir + '/' + mem
        Reader_prediction(path)


def cropImages(y, x, h, w):
    for i in range(len(dataFrame['image'])):
        dataFrame['image'][i] = dataFrame['image'][i][y:y + h, x:x + w]


def preprocess_images():
    for i in range(len(dataFrame['image'])):
        print("Processing img " + str(i+1))

        image = np.array(dataFrame['image'][i])
        img = dataFrame['image'][i]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary_m = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19, 25)
        colorized_from_bin = img

        rows, cols, third = img.shape

        for j in range(rows):
            for k in range(cols):
                colorized_from_bin.itemset((j, k, 0), binary_m[j, k])
                colorized_from_bin.itemset((j, k, 1), binary_m[j, k])
                colorized_from_bin.itemset((j, k, 2), binary_m[j, k])

        output_image = colorized_from_bin
        return output_image


# save predictions separately for the digit recognition model
# TODO the annotations for the DRM are currently done with hand-cropped predictions, it should be possible to crop a prediction automatically to extract it
def save_predictions(predictions):
    dir = r'E:\Programs\PyCharmProjects\Python\Machine Learning Projects\digit_recognition_model\Input_cropped\predictions'
    os.chdir(dir)
    print("Operation starting: saving predictions")

    for i in range(len(predictions)):
        pred_n = cv2.normalize(predictions[i], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        hsv = cv2.cvtColor(pred_n, cv2.COLOR_BGR2HSV)
        lower_boundary = np.array([50, 100, 200])
        upper_boundary = np.array([70, 255, 255])
        hsv_mask = cv2.inRange(hsv, lower_boundary, upper_boundary)
        serial = cv2.bitwise_and(pred_n, pred_n, mask=hsv_mask)

        cv2.imshow("prediction", predictions[i])
        cv2.imshow("normalized", pred_n)
        cv2.imshow("hsv", hsv)
        cv2.imshow("serial", serial)

        serial[:, :, 0] = 0
        serial[:, :, 2] = 0

        cv2.imshow("serial normalized", serial)

        im = serial
        na = np.array(im)
        orig = na.copy()  # Save original
        # im = im.filter(ImageFilter.MedianFilter(3))

        cv2.imshow("im", im)
        cv2.imshow("na", na)
        cv2.imshow("orig", orig)
        cv2.imshow("na", na)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        greenY, greenX = np.where(np.all(na == [0, 255, 0], axis=2))

        top, bottom = greenY[0], greenY[-1]
        left, right = greenX[0], greenX[-1]
        padding = 0
        ROI = orig[top+padding:bottom-padding, left-padding:right+padding]
        serial_cropped = Image.fromarray(ROI)

        filename = 'filtered_serial' + str(i + 1) + '.png'
        # result = cv2.imwrite(filename, serial_cropped)
        result = serial_cropped.save(filename)
        if result:
            print("File saved successfully")
        else:
            print("Error in saving file")

    print("Operation ended: saving predictions")


# function for getting 16 predictions
def predict_n(valMap, model, size=16, shape=256):
    # getting and proccessing val data
    img = valMap['image']
    # mask = valMap['box']
    # mask = mask[0:size]

    imgProc = img[0:size]
    imgProc = np.array(img)

    predictions = model.predict(imgProc)

    # needs 3 layers (see "output =" in GiveMeUnet())
    for i in range(len(predictions)):
        predictions[i] = cv2.merge((predictions[i, :, :, 0], predictions[i, :, :, 1], predictions[i, :, :, 2]))

    return predictions, imgProc
    # return predictions, imgProc, mask


dataFrame = {
    'image': [],
    'box': []
}

# load images
# Iterator('Input_left_190/Images/color')
Iterator('Input_left_190/Images/recolorized')

# crop images
box_h, box_w = 256, 416
box_h, box_w = 256, 416
cropImages(100, 840, box_h, box_w)

# prep images
# preprocess_images()

# load model
folder = 'TransportNumberRec_20210728_134841'
location = 'Models/' + folder + '/model'

print('loading model at: ' + location)
reconstructed_model = tf.keras.models.load_model(location)

print('loading succesful')

print('model summary')
reconstructed_model.summary()
reconstructed_model.get_weights()
reconstructed_model.optimizer

# predict random image
predictions, actuals = predict_n(dataFrame, reconstructed_model, len(dataFrame['image']))

save_predictions(predictions)