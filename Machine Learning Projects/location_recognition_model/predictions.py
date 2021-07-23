import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import numpy as np
import os
from time import perf_counter
import cv2
from tensorflow import keras

# def Reader(path, annotPath, size_x=1700, size_y=2338, isPlot=False, showTime=False):
#     start = perf_counter()
#     img = plt.imread(path)
#     img = np.array(img[:, :, 0:3])
#
#     if isPlot:
#         plt.subplot(1, 2, 1)
#         plt.imshow(img)
#
#     # parsing html
#     with open(annotPath, 'r') as f:
#         data = f.read()
#
#     Bs = BeautifulSoup(data, "html.parser")
#     xMax = int(Bs.find('x').text) + int(Bs.find('width').text)
#     xMin = int(Bs.find('x').text)
#     yMax = int(Bs.find('y').text) + int(Bs.find('height').text)
#     yMin = int(Bs.find('y').text)
#     # print(xMin, xMax, yMin, yMax, img.shape)
#     # storing data
#     dataFrame['image'].append(cv2.resize(img, (size_x, size_y)))
#     # drawing box
#     imgBoxed = cv2.rectangle(img, (xMin, yMin), (xMax, yMax), (0 ,255, 0), 2)
#     # storing data into frame
#
#     x = yMin
#     while x < yMax:
#         y = xMin
#         while y < xMax:
#             img[x, y, 1] = 1
#             y += 1
#         x += 1
#
#     dataFrame['box'].append(cv2.resize(img, (size_x, size_y)))
#
#     if isPlot:
#         plt.subplot(1, 2, 2)
#         plt.imshow(cv2.resize(imgBoxed, (size_x, size_y)))
#
#     if showTime:
#         return perf_counter() - start


# def Iterator(imageDir, annotDir):
#     start = perf_counter()
#     imageNames = os.listdir(imageDir)
#
#     for mem in imageNames:
#         path = imageDir + '/' + mem
#         annotPath = annotDir + '/' + mem.split('.')[0] + '.xml'
#         Reader(path, annotPath)
#
#     return perf_counter() - start


# read cropped images, without adding boxes
def read_cropped_images(imageDir):
    imageNames = os.listdir(imageDir)

    for num in imageNames:
        path = imageDir + '/' + num
        # Reader(path)
        img = plt.imread(path)
        dataFrame['images'].append(img)


def cropImages(y, x, h, w):

    counter = 0
    for image in dataFrame['box']:
        dataFrame['box'][counter] = image[y:y + h, x:x + w]
        counter += 1

    counter = 0
    for image in dataFrame['image']:
        dataFrame['image'][counter] = image[y:y + h, x:x + w]
        counter += 1


def accuracy_loss_graphs(retVal, show_graphs):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Accuracy')
    plt.plot(retVal.history['accuracy'], label='training_accuracy')
    plt.plot(retVal.history['val_accuracy'], label='validation_accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.title('Loss values')
    plt.plot(retVal.history['loss'], label='training_loss')
    plt.plot(retVal.history['val_loss'], label='validaton_loss')
    plt.legend()
    plt.grid(True)

    if show_graphs:
        plt.show()


## function for getting 16 predictions
def predict16(valMap, model, shape=256):
    ## getting and proccessing val data
    img = valMap['image']
    mask = valMap['box']
    mask = mask[0:16]

    imgProc = img[0:16]
    imgProc = np.array(img)

    predictions = model.predict(imgProc)
    for i in range(len(predictions)):
        predictions[i] = cv2.merge((predictions[i, :, :, 0], predictions[i, :, :, 1], predictions[i, :, :, 2]))

    return predictions, imgProc, mask


def Plotter(img, predMask, groundTruth):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('original Image')

    ## Adding Image sharpening step here
    ## it is a sharpening filter
    filter = np.array([[-1, -1, -1], [-1, 8.99, -1], [-1, -1, -1]])
    # filter = np.array([[0.1, -1, 0.1], [-1, 5, -1], [0.1, -1, 0.1]])
    # filter = np.array([[-1, 0.1, -1], [0.1, 5, 0.1], [-1, 0.1, -1]])
    imgSharpen = cv2.filter2D(predMask, -1, filter)

    plt.subplot(1, 3, 2)
    plt.imshow(imgSharpen)
    plt.title('Predicted Box position')

    plt.subplot(1, 3, 3)
    plt.imshow(groundTruth)
    plt.title('actual box Position')
    plt.show()


def ShowCroppedSerials(cropped_images):
    fraction = 9
    fraction_img_count = int(len(cropped_images)/fraction)

    for i in range(fraction):
        plt.figure(figsize=(16, 8))
        plt.autoscale()
        plt.tight_layout()

        for j in range(fraction_img_count):
            plt.subplot(6, 4, j+1)
            plt.imshow(cropped_images[j+fraction_img_count*i])
            plt.title('img:' + str(j+fraction_img_count*i))
            plt.axis('off')

        plt.show()


dataFrame = {
    'images' : []
}

read_cropped_images('Input_left_190/Images_cropped')
# ShowCroppedSerials(dataFrame['images'])

# Load a saved model
# trained_model_h5 = keras.models.load_model('E:\Programs\PyCharmProjects\Python\Machine Learning Projects\location_recognition_model\Models\TransportNumberRec_20210714_150709\model\saved_model.h5')
trained_model_pb = keras.models.load_model('E:\Programs\PyCharmProjects\Python\Machine Learning Projects\location_recognition_model\Models\TransportNumberRec_20210714_150709\model\saved_model.pb')
trained_model = trained_model_pb

accuracy_loss_graphs(trained_model, True)

# sixteenPrediction, actuals, masks = predict16(dataFrame, trained_model)
#
# Plotter(actuals[1], sixteenPrediction[1], masks[1])
# Plotter(actuals[2], sixteenPrediction[2], masks[2])
# Plotter(actuals[3], sixteenPrediction[3], masks[3])
# Plotter(actuals[4], sixteenPrediction[4], masks[4])
