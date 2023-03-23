import tensorflow as tf
from tensorflow.keras import models, layers
from astrooop import Process_fake
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
import numpy as np

with open('2000gal_1000star_3000_3000.pkl', 'rb') as f:
    test = pickle.load(f)

labeldata = test.createdataset()
print(labeldata['data'].shape) 
train_labels, test_labels, train_images, test_images = train_test_split(labeldata['label'], labeldata['data'], test_size=300,train_size=1000)

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)