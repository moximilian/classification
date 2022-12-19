import numpy as np
import pandas as pd
import cv2 as cv
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import sys
import time
import tensorflow as tf
import re
import keras.applications.mobilenet_v2 as mobilenetv2

from tensorflow import keras
from PIL import Image
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Input, GlobalAveragePooling2D
from keras.layers.preprocessing import normalization
from keras.models import Model, Sequential
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.layers import Lambda
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.utils import image_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.applications import imagenet_utils

import warnings
warnings.filterwarnings('ignore')
def mobilenetv2_preprocessing(img):
    return mobilenetv2.preprocess_input(img)


def add_class_name_prefix(df, col_name):
    df[col_name] = df[col_name].apply(lambda x: x[:re.search("\d", x).start()] + '/' + x)
    return df



IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

base_path = "D:/3 семестр/СТруктура/project_1.0/garbage_classification/"

categories = {0: 'battery', 1: 'biological', 2: 'brown-glass', 3: 'cardboard', 4: 'clothes', 5: 'green-glass',
              6: 'metal', 7: 'paper', 8: 'plastic', 9: 'shoes', 10: 'trash',
              11: 'white-glass'}

filenames_list = []
categories_list = []

for category in categories:
    filenames = os.listdir(base_path + categories[category])

    filenames_list = filenames_list + filenames
    categories_list = categories_list + [category] * len(filenames)

df = pd.DataFrame({
    'filename': filenames_list,
    'category': categories_list
})

df = add_class_name_prefix(df, 'filename')

df = df.sample(frac=1).reset_index(drop=True)


df.head()

df_visualization = df.copy()
df_visualization['category'] = df_visualization['category'].apply(lambda x: categories[x])

df_visualization['category'].value_counts().plot.bar(x='count', y='category')

plt.xlabel("Garbage Classes", labelpad=14)
plt.ylabel("Images Count", labelpad=14)

plt.title("Count of images per class", y=1.02)

mobilenetv2_layer = mobilenetv2.MobileNetV2(include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
                                            weights='D:/structure_app/neural_network'
                                                    '/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5')

mobilenetv2_layer.trainable = False

model = Sequential()
model.add(keras.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))

model.add(Lambda(mobilenetv2_preprocessing))

model.add(mobilenetv2_layer)
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(Dense(len(categories), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])


df["category"] = df["category"].replace(categories)

train_df, validate_df = train_test_split(df, test_size=0.2, random_state=42)
validate_df, test_df = train_test_split(validate_df, test_size=0.5, random_state=42)

train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]

model.load_weights("D:/structure_app/neural_network/model12.h5")

img = cv.imread("D:/structure_app/recognise_pic.jpg")
resized = cv.resize(img, (224, 224))
img = cv.cvtColor(resized, cv.COLOR_BGR2RGB)

x = image_utils.img_to_array(img)
x = np.expand_dims(x, axis=0)
result = model.predict(x)
CURSOR_UP = '\033[F'
ERASE_LINE = '\033[K'
print(CURSOR_UP + ERASE_LINE)
output_neuron = np.argmax(result[0])
my_file = open("D:/structure_app/network_result.txt", "w+")
if categories[output_neuron] == "battery":
    my_file.write("Батарейка")
if categories[output_neuron] == "biological" or categories[output_neuron] == "shoes" or \
        categories[output_neuron] == "trash":
    my_file.write("Не определено")
if categories[output_neuron] == "brown-glass" or categories[output_neuron] == "green-glass" or \
        categories[output_neuron] == "white-glass":
    my_file.write("Стекло")
if categories[output_neuron] == "cardboard" or categories[output_neuron] == "paper":
    my_file.write("Бумага")
if categories[output_neuron] == "metal":
    my_file.write("Металл")
if categories[output_neuron] == "plastic":
    my_file.write("Пластик")
my_file.close()
print(categories[output_neuron])
