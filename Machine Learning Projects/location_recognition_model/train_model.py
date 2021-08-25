# import np and tf and set seeds before anything else
import warnings

np_seed = 10
tf_seed = 20
import numpy as np

np.random.seed(np_seed)
import tensorflow as tf

tf.random.set_seed(tf_seed)

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

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

    path_arr = path.split('/')
    path_arr = path_arr[-1].split('.')
    cmr_nr = path_arr[-2]

    preprocess_single_image(img, cmr_nr, False, True)

    if isPlot:
        plt.subplot(1, 2, 1)
        plt.imshow(img)

    # parsing html
    dir = r'E:\Programs\PyCharmProjects\Python\Machine Learning Projects\location_recognition_model'
    os.chdir(dir)
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
    # imgBoxed = cv2.rectangle(img, (xMin, yMin), (xMax, yMax), (0, 255, 0), 2)  # color
    # imgBoxed = cv2.rectangle(img, (xMin, yMin), (xMax, yMax), 0, 2)  # greyscale/binary

    # storing data into frame (displayed as purple box)
    x = yMin
    while x < yMax:
        y = xMin
        while y < xMax:
            img[x, y, 1] = 1
            y += 1
        x += 1

    dataFrame['box'].append(cv2.resize(img, (size_x, size_y)))
    # mask = cv2.resize(img, (size_x, size_y))
    # dataFrame['box'].append(mask[:, :, 0])

    # if isPlot:
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(cv2.resize(imgBoxed, (size_x, size_y)))
    #     plt.show()

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


# deprecated, moved to preprocess_single_images due to memory usage.
# also outdated, do not use
def preprocess_images(show_plots=False):
    warnings.warn("preprocess_images is deprecated, use preprocess_single_image instead", DeprecationWarning)

    for i in range(len(dataFrame['image'])):
        img = dataFrame['image'][i]
        dataFrame['image'][i] = cv2.cvtColor(dataFrame['image'][i], cv2.COLOR_BGR2GRAY)

        # rgb2lab, works
        # dataFrame['image'][i] = rgb2lab(dataFrame['image'][i])
        # dataFrame['image'][i][..., 1] = dataFrame['image'][i][..., 2] = 0
        # dataFrame['image'][i] = lab2rgb(dataFrame['image'][i])

        # rgb2gray, in dev
        # dataFrame['image'][i] = rgb2gray(dataFrame['image'][i])
        # dataFrame['image'][i][..., 1] = dataFrame['image'][i][..., 2] = 0
        # dataFrame['image'][i] = lab2rgb(dataFrame['image'][i])

        grey = dataFrame['image'][i]

        # cv2.invert(dataFrame['image'][i])
        # _, dataFrame['image'][i] = cv2.threshold(dataFrame['image'][i], 127, 255, cv2.THRESH_BINARY)
        dataFrame['image'][i] = cv2.adaptiveThreshold(dataFrame['image'][i], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19, 25)
        # _, dataFrame['image'][i] = cv2.threshold(dataFrame['image'][i], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        binary = dataFrame['image'][i]

        if show_plots:
            plt.subplot(1, 3, 1)
            plt.title('Original')
            plt.imshow(img)

            plt.subplot(1, 3, 2)
            plt.title('Greyscale')
            plt.imshow(grey, cmap="gray")

            plt.subplot(1, 3, 3)
            plt.title('Binary')
            plt.imshow(binary, cmap="gray")
            plt.show()


def preprocess_single_image(image, cmr_nr, show_plots=False, save_images=False):
    print("Processing img " + cmr_nr)

    img = image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # _, binary_threshold63 = cv2.threshold(gray, 63, 255, cv2.THRESH_BINARY)
    # _, binary_threshold127 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # _, binary_threshold191 = cv2.threshold(gray, 191, 255, cv2.THRESH_BINARY)
    binary_m = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19, 25)
    # binary_g = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 25)

    colorized_from_bin = img

    rows, cols, third = img.shape

    for i in range(rows):
        for j in range(cols):
            colorized_from_bin.itemset((i, j, 0), binary_m[i, j])
            colorized_from_bin.itemset((i, j, 1), binary_m[i, j])
            colorized_from_bin.itemset((i, j, 2), binary_m[i, j])

    output_image = colorized_from_bin

    if save_images:
        filename = cmr_nr + '.png'

        dir = r'E:\Programs\PyCharmProjects\Python\Machine Learning Projects\location_recognition_model\Input_left_190\Images\grayscale'
        os.chdir(dir)
        result = cv2.imwrite(filename, gray)
        if result:
            print("Image saved successfully saved in grayscale")
        else:
            print("Error in saving grayscale image")

        # # only used to showcase THRESH_BINARY
        # dir = r'E:\Programs\PyCharmProjects\Python\Machine Learning Projects\location_recognition_model\Input_left_190\Images\binary_threshold63'
        # os.chdir(dir)
        # result = cv2.imwrite(filename, binary_threshold63)
        # if result:
        #     print("Image saved successfully saved in binary colors to folder binary_threshold63")
        # else:
        #     print("Error in saving binary image")
        #
        # # only used to showcase THRESH_BINARY
        # dir = r'E:\Programs\PyCharmProjects\Python\Machine Learning Projects\location_recognition_model\Input_left_190\Images\binary_threshold127'
        # os.chdir(dir)
        # result = cv2.imwrite(filename, binary_threshold127)
        # if result:
        #     print("Image saved successfully saved in binary colors to folder binary_threshold127")
        # else:
        #     print("Error in saving binary image")
        #
        # # only used to showcase THRESH_BINARY
        # dir = r'E:\Programs\PyCharmProjects\Python\Machine Learning Projects\location_recognition_model\Input_left_190\Images\binary_threshold191'
        # os.chdir(dir)
        # result = cv2.imwrite(filename, binary_threshold191)
        # if result:
        #     print("Image saved successfully saved in binary colors to folder binary_threshold191")
        # else:
        #     print("Error in saving binary image")

        dir = r'E:\Programs\PyCharmProjects\Python\Machine Learning Projects\location_recognition_model\Input_left_190\Images\binarized'
        os.chdir(dir)
        filename = cmr_nr + '.png'
        if result:
            print("Image saved successfully saved in binary colors")
        else:
            print("Error in saving binary image")

        dir = r'E:\Programs\PyCharmProjects\Python\Machine Learning Projects\location_recognition_model\Input_left_190\Images\recolorized'
        os.chdir(dir)
        result = cv2.imwrite(filename, colorized_from_bin)
        if result:
            print("Image saved successfully saved in color")
        else:
            print("Error in saving color image")

    if show_plots:
        plt.subplot(1, 4, 1)
        plt.title('Original')
        plt.imshow(img)

        plt.subplot(1, 4, 2)
        plt.title('Greyscale')
        plt.imshow(gray, cmap="gray")

        plt.subplot(1, 4, 3)
        plt.title('Binary_MEAN')
        plt.imshow(binary_m, cmap="gray")

        plt.subplot(1, 4, 4)
        plt.title('Colorized')
        plt.imshow(colorized_from_bin)
        plt.show()

        #second plot
        plt.subplot(1, 3, 1)
        plt.title('binary thresh 60')
        plt.imshow(binary_threshold60, cmap="gray")

        plt.subplot(1, 3, 2)
        plt.title('binary thresh 127')
        plt.imshow(binary_threshold127, cmap="gray")

        plt.subplot(1, 3, 3)
        plt.title('binary thresh 191')
        plt.imshow(binary_threshold191, cmap="gray")
        plt.show()

    return output_image


# def segment_prediction(image, cmr_nr, show_plots=False, save_images=False):
#     print("Processing prediction of CMR " + cmr_nr)
#
#     img = image
#
#     rows, cols, third = img.shape
#
#     for i in range(rows):
#         for j in range(cols):
#             b = image[i, j, 0]
#             g = image[i, j, 1]
#             r = image[i, j, 2]
#
#
#     output_image = cv2.merge([b, g, r])
#
#     if save_images:
#         dir = r'E:\Programs\PyCharmProjects\Python\Machine Learning Projects\location_recognition_model\output'
#         os.chdir(dir)
#         filename = cmr_nr + '.png'
#         result = cv2.imwrite(filename, output_image)
#         if result:
#             print("Image saved successfully")
#         else:
#             print("Error in saving image")
#
#     if show_plots:
#         plt.subplot(1, 2, 1)
#         plt.title('Original')
#         plt.imshow(img)
#
#         plt.subplot(1, 2, 2)
#         plt.title('Segmented')
#         plt.imshow(output_image)
#
#     return output_image


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

    output = tf.keras.layers.Conv2D(3, (1, 1), activation='sigmoid')(c9)
    model = tf.keras.Model(inputs=[inputImage], outputs=[output])
    return model


def show_original_and_annotation():
    plt.subplot(1, 2, 1)
    plt.imshow(dataFrame['image'][0])
    plt.subplot(1, 2, 2)
    plt.imshow(dataFrame['box'][0])
    plt.show()


def show_hsv():
    predMask = dataFrame['box'][0]
    img = dataFrame['image'][0]

    img_n = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    pred_n = cv2.normalize(predMask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    hsv = cv2.cvtColor(pred_n, cv2.COLOR_BGR2HSV)
    lower_boundary = np.array([55, 190, 255])
    upper_boundary = np.array([65, 255, 255])
    hsv_mask = cv2.inRange(hsv, lower_boundary, upper_boundary)
    new_img = cv2.bitwise_and(pred_n, pred_n, mask=hsv_mask)

    cv2.imshow("img", img)
    cv2.imshow("img n", img_n)
    cv2.imshow("box", predMask)
    cv2.imshow("box n", pred_n)
    cv2.imshow("hsv", hsv)
    cv2.imshow("hsv mask", hsv_mask)
    cv2.imshow("new image", new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# save predictions separately for the digit recognition model
def save_predictions(predictions):
    dir = r'E:\Programs\PyCharmProjects\Python\Machine Learning Projects\digit_recognition_model\Input_cropped\predictions\cropped'
    os.chdir(dir)
    print("Operation starting: saving predictions")

    for i in range(len(predictions)):
        pred_n = cv2.normalize(predictions[i], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        hsv = cv2.cvtColor(pred_n, cv2.COLOR_BGR2HSV)
        lower_boundary = np.array([55, 190, 255])
        upper_boundary = np.array([65, 255, 255])
        hsv_mask = cv2.inRange(hsv, lower_boundary, upper_boundary)

        coords = np.argwhere(hsv_mask)
        whiteY, whiteX = zip(*coords)
        whiteY = np.sort(whiteY)
        whiteX = np.sort(whiteX)
        top, bottom = whiteY[0], whiteY[-1]
        left, right = whiteX[0], whiteX[-1]
        padding = 8
        serial_cropped = hsv_mask[top - padding:bottom + padding, left - padding:right + padding]
        serial_cropped_inv = cv2.bitwise_not(serial_cropped)

        filename = 'pred_serial' + str(i + 1) + '.png'
        result = cv2.imwrite(filename, serial_cropped_inv)
        if result:
            print("File saved successfully")
        else:
            print("Error in saving file")

    print("Operation ended: saving predictions")


def predict_n(valMap, model, size=16, shape=256):
    # getting and proccessing val data
    img = valMap['image']
    mask = valMap['box']
    mask = mask[0:size]

    imgProc = img[0:size]
    imgProc = np.array(img)

    predictions = model.predict(imgProc)

    # needs 3 layers (see "output =" in GiveMeUnet())
    for i in range(len(predictions)):
        predictions[i] = cv2.merge((predictions[i, :, :, 0], predictions[i, :, :, 1], predictions[i, :, :, 2]))

    return predictions, imgProc, mask


def Plotter(img, predMask, groundTruth, show_predictions=False) -> object:
    plt.figure(figsize=(8, 4))

    plt.subplot(2, 3, 1)
    plt.title('Original image')
    plt.imshow(img)

    # ## sharpen image
    # filter = np.array([[-1, -1, -1], [-1, 8.99, -1], [-1, -1, -1]])
    # imgSharpen = cv2.filter2D(predMask, -1, filter)

    plt.subplot(2, 3, 2)
    plt.title('Annotation box')
    plt.imshow(groundTruth)

    plt.subplot(2, 3, 3)
    plt.title('Predicted serial')
    plt.imshow(predMask)

    pred_n = cv2.normalize(predMask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    hsv = cv2.cvtColor(pred_n, cv2.COLOR_BGR2HSV)
    lower_boundary = np.array([55, 200, 210])
    upper_boundary = np.array([65, 255, 255])
    hsv_mask = cv2.inRange(hsv, lower_boundary, upper_boundary)

    coords = np.argwhere(hsv_mask)
    whiteY, whiteX = zip(*coords)
    whiteY = np.sort(whiteY)
    whiteX = np.sort(whiteX)
    top, bottom = whiteY[0], whiteY[-1]
    left, right = whiteX[0], whiteX[-1]
    padding = 8
    serial_cropped = hsv_mask[top - padding:bottom + padding, left - padding:right + padding]
    serial_cropped_inv = cv2.bitwise_not(serial_cropped)

    plt.subplot(2, 3, 4)
    plt.title('HSV of Predmask')
    plt.imshow(hsv)

    plt.subplot(2, 3, 5)
    plt.title('HSV range mask of prediction')
    plt.imshow(hsv_mask)

    plt.subplot(2, 3, 6)
    plt.title('final image (inverted)')
    plt.imshow(serial_cropped_inv)

    plt.savefig('Models/TransportNumberRec_' + training_date + '/prediction.png')

    if show_predictions:
        plt.show()


# show images in a grid. Used to find out the correct crop settings
def ShowCroppedSerials(cropped_images):
    fraction = 9
    fraction_img_count = int(len(cropped_images) / fraction)

    for i in range(fraction):
        plt.figure(figsize=(16, 8))
        plt.autoscale()
        plt.tight_layout()

        for j in range(fraction_img_count):
            plt.subplot(6, 4, j + 1)
            plt.imshow(cropped_images[j + fraction_img_count * i], cmap="gray")
            plt.title('img:' + str(j + fraction_img_count * i))
            plt.axis('off')

        plt.show()


def SaveCroppedSerials(cropped_images):
    dir = r'E:\Programs\PyCharmProjects\Python\Machine Learning Projects\digit_recognition_model\Input_cropped\binary'
    os.chdir(dir)
    for i in range(len(cropped_images)):
        cv2.imshow(cropped_images[i])
        filename = 'cropped_cmr' + str(i+1) + '.png'
        noise_image_norm = cv2.normalize(cropped_images[i], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        result = cv2.imwrite(filename, noise_image_norm)
        if result:
            print("File saved successfully")
        else:
            print("Error in saving file")

        cv2.imshow(noise_image_norm)


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
# recolorized images used because model requires three color channels,
# and it's faster to use preprocessed images, rather then repeating this process every time
os.chdir(r'E:\Programs\PyCharmProjects\Python\Machine Learning Projects\location_recognition_model')
Iterator('Input_left_190/Images/recolorized', 'Input_left_190/Annotations')
# Iterator_single('Input_left_190/Images/color', 'Input_left_190/Annotations')  # requires preprocess_single_image()

# full serial box
# box_w, box_h = 768, 256

# left side serial box
box_h, box_w = 256, 416
box_h, box_w = 256, 416

cropImages(100, 840, box_h, box_w)
# preprocess_images()

# ShowCroppedSerials(dataFrame['image'])
# SaveCroppedSerials(dataFrame['image'])

# displaying data loaded by our function
# show_original_and_annotation()
# show_hsv()

# settings used
training_dropouts = 0
training_test_size = 0.22
training_random_state = 22
training_epochs = 3
training_batch_size = 16

training_date = datetime.now().strftime("%Y%m%d_%H%M%S")
# training_date = '20210728_134841'

# split data in train and test data
x_train, x_test, y_train, y_test = train_test_split(np.array(dataFrame['image']), np.array(dataFrame['box']), test_size=training_test_size, random_state=training_random_state)

# train model
inputs = tf.keras.layers.Input((box_h, box_w, 3))
# inputs = tf.keras.layers.Input((box_h, box_w))
myTransformer = GiveMeUnet(inputs, dropouts=training_dropouts)
myTransformer.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
# tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss', min_delta=0, patience=3, verbose=0,
#     mode='auto', baseline=None, restore_best_weights=False
# )
retVal = myTransformer.fit(x_train, y_train, validation_data=(x_test, y_test), verbose=1, epochs=training_epochs, batch_size=training_batch_size)

#load model
# folder = 'TransportNumberRec_20210728_134841'
# location = 'Models/' + folder + '/model'

# print('loading model at: ' + location)
# reconstructed_model = tf.keras.models.load_model(location)
# myTransformer = reconstructed_model


# # save the model and the settings
myTransformer.save('Models/TransportNumberRec_' + training_date + '/model')
WriteUsedSettingsToFileInJson()

# reconstructed_model = tf.keras.models.load_model('Models/TransportNumberRec_' + training_date + '/model')
#
# np.testing.assert_allclose(
#     myTransformer.predict(x_train), reconstructed_model.predict(x_train)
# )

# make accuracy and loss in two graphs, boolean to show them
accuracy_loss_graphs(retVal, False)

# predict n images from the dataFrame
predictions, actuals, masks = predict_n(dataFrame, myTransformer)
# predict a serial, boolean to show

Plotter(actuals[7], predictions[7], masks[7])

# save_predictions(predictions)