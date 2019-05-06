from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')

print(tf.__version__)

labels, data, test_data = [], [], []


for line in open('federalist_papers_data.csv'):
    line = line.split(',')
    
    if int(line[0]) == 3:
        test_data.append(line[1:])
    else:
        labels.append(int(line[0]) - 1)
        data.append(line[1:])
    
labels = np.array(labels)
data = np.array(data)
test_data = np.array(test_data)
    
# Build a model
model = keras.Sequential([
    keras.layers.Dense(70),
    keras.layers.Dense(128, activation=tf.nn.sigmoid),
    keras.layers.Dense(128, activation=tf.nn.sigmoid),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

# Compile the model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(data, labels, epochs=30)

predictions = model.predict(test_data)
hamilton, madison = 0, 0

for disputed_paper in predictions:
    if disputed_paper[0] > disputed_paper[1]:
        hamilton += 1
    else:
        madison += 1

print(hamilton, "of 12 papers were written by Hamilton and", madison, "of the 12 papers were written by Madison!")


