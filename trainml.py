import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras import models, layers
from astroop3 import Process_frompkl
from astroop3 import Process_fake
import numpy as np


test = Process_frompkl('TrainingData2')

labeldata = test.createdataset()
print(labeldata['data'].shape) 
train_labels, test_labels, train_images, test_images = train_test_split(labeldata['label'], labeldata['data'], test_size=300,train_size=1500)

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
model.add(layers.AveragePooling2D(2,2))
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
plt.grid()
plt.show()

#model = models.load_model('TrainedModel2.h5')
probability_model = models.Sequential([model, tf.keras.layers.Softmax()])
predicts = probability_model.predict_on_batch(test_images)

predicts = np.where(predicts[:,1] > 0.5, 1, 0)

print(classification_report(test_labels, predicts))
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

model.save('TrainedModel2.h5')
