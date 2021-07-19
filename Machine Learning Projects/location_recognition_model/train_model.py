# import np and tf and set seeds before anything else
import imageio

np_seed = 10
tf_seed = 20
import numpy as np

np.random.seed(np_seed)
import tensorflow as tf

tf.random.set_seed(tf_seed)

import json
import time
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import pandas as pd
import os
from time import perf_counter
import cv2
from keras.models import load_model
from sklearn.model_selection import train_test_split
from datetime import datetime


# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

# def limitgpu(maxmem):
# 	gpus = tf.config.list_physical_devices('GPU')
# 	if gpus:
# 		# Restrict TensorFlow to only allocate a fraction of GPU memory
# 		try:
# 			for gpu in gpus:
# 				tf.config.experimental.set_virtual_device_configuration(gpu,
# 						[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=maxmem)])
# 		except RuntimeError as e:
# 			# Virtual devices must be set before GPUs have been initialized
# 			print(e)
#
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
# # 1.5GB
# limitgpu(1024+512)


def Reader(path, annotPath, size_x=1700, size_y=2338, isPlot=False, showTime=False):  # offset = 50, 882
    start = perf_counter()
    img = plt.imread(path)
    img = np.array(img[:, :, 0:3])

    if isPlot:
        plt.subplot(1, 2, 1)
        plt.imshow(img)

    # parsing html
    with open(annotPath, 'r') as f:
        data = f.read()

    Bs = BeautifulSoup(data, "html.parser")
    xMax = int(Bs.find('x').text) + int(Bs.find('width').text)
    xMin = int(Bs.find('x').text)
    yMax = int(Bs.find('y').text) + int(Bs.find('height').text)
    yMin = int(Bs.find('y').text)
    # print(xMin, xMax, yMin, yMax, img.shape)
    # storing data
    dataFrame['image'].append(cv2.resize(img, (size_x, size_y)))
    # drawing box
    imgBoxed = cv2.rectangle(img, (xMin, yMin), (xMax, yMax), (0, 255, 0), 2)
    # storing data into frame

    x = yMin
    while x < yMax:
        y = xMin
        while y < xMax:
            img[x, y, 1] = 1
            y += 1
        x += 1

    # dataFrame['box'].append(cv2.resize(img, (size_x, size_y)))
    mask = cv2.resize(img, (size_x, size_y))
    dataFrame['box'].append(mask[:, :, 1])

    if isPlot:
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.resize(imgBoxed, (size_x, size_y)))

    if showTime:
        return perf_counter() - start


def Iterator(imageDir, annotDir):
    start = perf_counter()
    imageNames = os.listdir(imageDir)

    for mem in imageNames:
        path = imageDir + '/' + mem
        annotPath = annotDir + '/' + mem.split('.')[0] + '.xml'
        Reader(path, annotPath)

    return perf_counter() - start


def cropImages(y, x, h, w):
    for i in range(len(dataFrame['image'])):
        dataFrame['box'][i] = dataFrame['box'][i][y:y + h, x:x + w]
        dataFrame['image'][i] = dataFrame['image'][i][y:y + h, x:x + w]


# defining autoencoder model
def Conv2dBlock(inputTensor, numFilters, kernelSize=3, doBatchNorm=True):
    # first Conv
    x = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(kernelSize, kernelSize),
                               kernel_initializer='he_normal', padding='same')(inputTensor)

    if doBatchNorm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation('relu')(x)

    # Second Conv
    x = tf.keras.layers.Conv2D(filters=numFilters, kernel_size=(kernelSize, kernelSize),
                               kernel_initializer='he_normal', padding='same')(x)
    if doBatchNorm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation('relu')(x)

    return x


# Now defining Unet
def GiveMeUnet(inputImage, numFilters=16, dropouts=0.1, doBatchNorm=True):
    # defining encoder Path
    c1 = Conv2dBlock(inputImage, numFilters * 1, kernelSize=3, doBatchNorm=doBatchNorm)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
    p1 = tf.keras.layers.Dropout(dropouts)(p1)

    c2 = Conv2dBlock(p1, numFilters * 2, kernelSize=3, doBatchNorm=doBatchNorm)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    p2 = tf.keras.layers.Dropout(dropouts)(p2)

    c3 = Conv2dBlock(p2, numFilters * 4, kernelSize=3, doBatchNorm=doBatchNorm)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
    p3 = tf.keras.layers.Dropout(dropouts)(p3)

    c4 = Conv2dBlock(p3, numFilters * 8, kernelSize=3, doBatchNorm=doBatchNorm)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)
    p4 = tf.keras.layers.Dropout(dropouts)(p4)

    c5 = Conv2dBlock(p4, numFilters * 16, kernelSize=3, doBatchNorm=doBatchNorm)

    # defining decoder path
    u6 = tf.keras.layers.Conv2DTranspose(numFilters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    u6 = tf.keras.layers.Dropout(dropouts)(u6)
    c6 = Conv2dBlock(u6, numFilters * 8, kernelSize=3, doBatchNorm=doBatchNorm)

    u7 = tf.keras.layers.Conv2DTranspose(numFilters * 4, (3, 3), strides=(2, 2), padding='same')(c6)

    u7 = tf.keras.layers.concatenate([u7, c3])
    u7 = tf.keras.layers.Dropout(dropouts)(u7)
    c7 = Conv2dBlock(u7, numFilters * 4, kernelSize=3, doBatchNorm=doBatchNorm)

    u8 = tf.keras.layers.Conv2DTranspose(numFilters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    u8 = tf.keras.layers.Dropout(dropouts)(u8)
    c8 = Conv2dBlock(u8, numFilters * 2, kernelSize=3, doBatchNorm=doBatchNorm)

    u9 = tf.keras.layers.Conv2DTranspose(numFilters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    u9 = tf.keras.layers.Dropout(dropouts)(u9)
    c9 = Conv2dBlock(u9, numFilters * 1, kernelSize=3, doBatchNorm=doBatchNorm)

    output = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = tf.keras.Model(inputs=[inputImage], outputs=[output])
    return model


## function for getting 16 predictions
def predict16(valMap, model, shape=256):
    ## getting and proccessing val data
    img = valMap['image']
    mask = valMap['box']
    mask = mask[0:16]

    imgProc = img[0:16]
    imgProc = np.array(img)

    predictions = model.predict(imgProc)
    chan0, chan1, chan2 = predictions, predictions, predictions

    for i in range(len(predictions)):
        # predictions[i] = cv2.merge((predictions[i, :, :, 0], predictions[i, :, :, 1], predictions[i, :, :, 2]))
        chan0[i] = cv2.merge(predictions[i, :, :, 0])
        chan1[i] = cv2.merge(predictions[i, :, :, 1])
        chan2[i] = cv2.merge(predictions[i, :, :, 2])

    return predictions, imgProc, mask, chan0, chan1, chan2


def Plotter(img, predMask, groundTruth, show_predictions) -> object:
    plt.figure(figsize=(12, 3))

    plt.subplot(1, 4, 1)
    plt.title('Original image')
    plt.imshow(img)

    # ## sharpen image
    # filter = np.array([[-1, -1, -1], [-1, 8.99, -1], [-1, -1, -1]])
    # imgSharpen = cv2.filter2D(predMask, -1, filter)

    plt.subplot(1, 4, 2)
    plt.title('Annotation box')
    plt.imshow(groundTruth)

    plt.subplot(1, 4, 3)
    plt.title('Predicted serial')
    plt.imshow(predMask)

    imh = predMask
    imh[imh < 0.5] = 0
    imh[imh > 0.5] = 1

    plt.subplot(1, 4, 4)
    plt.title('Segmented image')
    plt.imshow(cv2.merge((imh, imh, imh)) * img)

    plt.savefig('Models/TransportNumberRec_' + training_date + '/prediction')

    if show_predictions:
        plt.show()


# def plotter_array(img, predMask, groundTruth, show_predictions):
#     plt.figure(figsize=(10, 5*len(img)))
#
#     for i in img:
#         plt
#         plt.subplot(i+1, 3, 1)
#         plt.imshow(img)
#         plt.title('original Image')
#
#         ## Adding Image sharpening step here
#         ## it is a sharpening filter
#         filter = np.array([[-1, -1, -1], [-1, 8.99, -1], [-1, -1, -1]])
#         # filter = np.array([[0.1, -1, 0.1], [-1, 5, -1], [0.1, -1, 0.1]])
#         # filter = np.array([[-1, 0.1, -1], [0.1, 5, 0.1], [-1, 0.1, -1]])
#         imgSharpen = cv2.filter2D(predMask, -1, filter)
#
#         plt.subplot(i+1, 3, 2)
#         plt.imshow(imgSharpen)
#         plt.title('Predicted Box position')
#
#         plt.subplot(i+1, 3, 3)
#         plt.imshow(groundTruth)
#         plt.title('actual box Position')
#         plt.savefig('Models/TransportNumberRec_' + training_date + '/prediction_array')
#
#     if show_predictions:
#         plt.show()


def ShowCroppedSerials(cropped_images):
    fraction = 9
    fraction_img_count = int(len(cropped_images) / fraction)

    for i in range(fraction):
        plt.figure(figsize=(16, 8))
        plt.autoscale()
        plt.tight_layout()

        for j in range(fraction_img_count):
            plt.subplot(6, 4, j + 1)
            plt.imshow(cropped_images[j + fraction_img_count * i])
            plt.title('img:' + str(j + fraction_img_count * i))
            plt.axis('off')

        plt.show()


def SaveCroppedSerials(cropped_images):
    dir = r'E:\Programs\PyCharmProjects\Python\Machine Learning Projects\digit_recognition_model\Input_cropped'
    # dir2 = os.path.dirname(__file__)
    # filename = os.path.join(dir, r'..\..\digit_recognition_model\Input_cropped')
    os.chdir(dir)
    # os.listdir(dir2)
    for i in range(len(cropped_images)):
        filename = 'cropped_cmr' + str(i+1) + '.jpeg'
        result = cv2.imwrite(filename, cropped_images[i])
        if result:
            print("File saved successfully")
        else:
            print("Error in saving file")


def WriteUsedSettingsToFileInJson():
    dict = {
        "image_height": 2338,
        "image_width": 1700,
        "box_height": box_h,
        "box_width": box_w,
        "epochs": training_epochs,
        "dropouts": training_dropouts,
        "test_size": training_test_size,
        "random_state": training_random_state,
        "batch_size": training_batch_size,
        "np_seed": np_seed,
        "tf_seed": tf_seed
    }

    with open('Models/TransportNumberRec_' + training_date + '/settings.json', 'w') as f:
        f.write(json.dumps(dict, indent=2))
        f.close()


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
    plt.savefig('Models/TransportNumberRec_' + training_date + '/plots')

    if show_graphs:
        plt.show()


dataFrame = {
    'image': [],
    'box': []
}

# CMR's with serial on the left half of the box (top right on the form)
Iterator('Input_left_190/Images', 'Input_left_190/Annotations')

# full serial box
# box_w, box_h = 768, 256

# left side serial box
box_h, box_w = 256, 416
box_h, box_w = 256, 416

cropImages(100, 840, box_h, box_w)

# ShowCroppedSerials(dataFrame['image'])
# SaveCroppedSerials(dataFrame['image'])

# displaying data loaded by our function
show_annotation_plot = False
if show_annotation_plot:
    plt.subplot(1, 2, 1)
    plt.imshow(dataFrame['image'][1])
    plt.subplot(1, 2, 2)
    plt.imshow(dataFrame['box'][1])
    plt.show()

# settings used
training_dropouts = 0
training_test_size = 0.22
training_random_state = 22
training_epochs = 5
training_batch_size = 8

training_date = datetime.now().strftime("%Y%m%d_%H%M%S")

# split data in train and test data
x_train, x_test, y_train, y_test = train_test_split(np.array(dataFrame['image']), np.array(dataFrame['box']), test_size=training_test_size, random_state=training_random_state)

# train model
inputs = tf.keras.layers.Input((box_h, box_w, 3))
myTransformer = GiveMeUnet(inputs, dropouts=training_dropouts)
myTransformer.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
retVal = myTransformer.fit(x_train, y_train, validation_data=(x_test, y_test), verbose=1, epochs=training_epochs, batch_size=training_batch_size)

# save the model and the settings
myTransformer.save('Models/TransportNumberRec_' + training_date + '/model')
WriteUsedSettingsToFileInJson()

# make accuracy and loss in two graphs, boolean to show them
# accuracy_loss_graphs(retVal, False)

# predict  16 images from the dataFrame (first 16, ordered ?)
sixteenPrediction, actuals, masks, chan0, chan1, chan2 = predict16(dataFrame, myTransformer)
# predict n serials, boolean to show
Plotter(actuals[1], sixteenPrediction[1][:, :, 0], masks[1], True)
Plotter(chan0[1], chan1[1], chan2[1], True)

# plotter_array(actuals[1:4], sixteenPrediction[1:4], masks[1:4], True)

