import numpy as np

import random
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plot

np.random.seed(42)
X = np.random.randint(0, 2, size=(100, 12)) 
Y = np.random.randint(0, 2, size=(100, 2))
np.savetxt('dataIn.txt', X, fmt='%d')
np.savetxt('dataOut.txt', Y, fmt='%d')

X = np.loadtxt('dataIn.txt')
Y = np.loadtxt('dataOut.txt')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

model = keras.Sequential([
 keras.layers.Dense(12, input_shape=(12,), activation='log_sigmoid'),
 keras.layers.Dense(2, activation='sigmoid')])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=50, batch_size=16, validation_data=(X_test, Y_test))

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(Y_test, axis=1)
print("Accuracy:", accuracy_score(y_true, y_pred))

plot.plot(history.history['loss'], label='Training loss')
plot.plot(history.history['val_loss'], label='Validation loss')
plot.xlabel('Epochs')
plot.ylabel('Loss')
plot.legend()
plot.title('Losses')
plot.show()

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

Y_train_labels = np.argmax(Y_train, axis=1)
Y_test_labels = np.argmax(Y_test, axis=1)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, Y_train_labels)
y_log_result = lr.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(Y_test_labels, y_log_result))

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, Y_train_labels)
y_rf_result = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(Y_test_labels, y_rf_result))