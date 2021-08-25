# import np and tf and set seeds before anything else
import keras
import numpy as np
import tensorflow as tf
import sys
import json
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import os
from time import perf_counter
# import cv2.cv2 as cv2
import cv2
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
    dir = r'E:\Programs\PyCharmProjects\Python\Machine Learning Projects\digit_recognition_model\Input_cropped\predictions\filtered'
    os.chdir(dir)
    print("Operation starting: saving predictions")
    print("amount of predictions: " + str(range(len(predictions))))
    counter = 0
    errors = []
    bad = [41, 59, 70, 76, 88, 98, 108, 184]

    for i in range(len(predictions)):
        pred_n = cv2.normalize(predictions[i], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        hsv = cv2.cvtColor(pred_n, cv2.COLOR_BGR2HSV)
        lower_boundary = np.array([55, 200, 210])
        upper_boundary = np.array([65, 255, 255])
        hsv_mask = cv2.inRange(hsv, lower_boundary, upper_boundary)

        dir = r'E:\Programs\PyCharmProjects\Python\Machine Learning Projects\digit_recognition_model\Input_cropped\predictions\hsv'
        os.chdir(dir)
        filename = 'hsv_' + str(i + 1) + '.png'
        cv2.imwrite(filename, hsv)

        # serial = cv2.bitwise_and(pred_n, pred_n, mask=hsv_mask)

        cleaned = hsv_mask

        # cv2.imshow("prediction", predictions[i])
        # cv2.imshow("hsv", hsv)
        # cv2.imshow("hsv mask", hsv_mask)

        im = cleaned
        med_5 = cv2.medianBlur(im, 5)
        med_3 = cv2.medianBlur(im, 3)
        med_5_3 = cv2.medianBlur(med_5, 3)

        # cv2.imshow("cleaned", cleaned)
        # cv2.imshow("med 3", med_3)
        # cv2.imshow("med 5", med_5)
        # cv2.imshow("med 5 3", med_5_3)

        coords = np.argwhere(im)
        whiteY, whiteX = zip(*coords)
        whiteY = np.sort(whiteY)
        whiteX = np.sort(whiteX)
        top, bottom = whiteY[0], whiteY[-1]
        left, right = whiteX[0], whiteX[-1]
        # print('top left:', str(left), str(top), 'bottom right:', str(right), str(bottom))
        padding = 8
        serial_cropped = im[top-padding:bottom+padding, left-padding:right+padding]
        # cv2.imshow("cropped serial", serial_cropped)
        serial_cropped_inv = cv2.bitwise_not(serial_cropped)
        # cv2.imshow("cropped serial inverted", serial_cropped_inv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        filename = 'filtered_serial' + str(i + 1) + '.png'
        try:
            dir = r'E:\Programs\PyCharmProjects\Python\Machine Learning Projects\digit_recognition_model\Input_cropped\predictions\cropped'
            os.chdir(dir)
            result = cv2.imwrite(filename, serial_cropped_inv)
            if result:
                print("File saved successfully: " + filename)
                counter += 1
            else:
                print("Error in saving file")
        except:
            errors.append("image " + str(i+1))
            errors.append(sys.exc_info())
            print("Oops!", sys.exc_info()[0], "occurred.")
            print("Next entry.")
            print()

    print("Operation ended: saving predictions")
    print("Operation ended: saved " + str(counter) + " images")

    print("Operation ended: The encountered errors were:")
    for i in errors:
        for j in i:
            print(j, end=" ")
        print()


# function for getting 16 predictions
def predict_n(valMap, model, size=16, shape=256):
    # getting and proccessing val data
    img = valMap['image']

    imgProc = img[0:size]
    imgProc = np.array(img)

    predictions = model.predict(imgProc)

    for i in range(len(predictions)):
        predictions[i] = cv2.merge((predictions[i, :, :, 0], predictions[i, :, :, 1], predictions[i, :, :, 2]))

    return predictions, imgProc
    # return predictions, imgProc, mask


dataFrame = {
    'image': []
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
# folder = 'TransportNumberRec_20210728_134841'  # 25 epoch
# folder = 'TransportNumberRec_20210728_225022'  # 50 epochs
folder = 'TransportNumberRec_20210729_001501'  # 50 epochs
location = 'Models/' + folder + '/model'

print('loading model at: ' + location)
reconstructed_model = tf.keras.models.load_model(location)
print('loading succesful')

# predict random image
predictions, actuals = predict_n(dataFrame, reconstructed_model, len(dataFrame['image']))

save_predictions(predictions)