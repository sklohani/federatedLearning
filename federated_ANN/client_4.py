## Import Modules
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import flwr as fl
from sklearn.preprocessing import MinMaxScaler
import scipy.io as sio
import sys

## Method to get test data
def getTestDataset():
    database = sio.loadmat('../testData/data_base_all_sequences_random.mat')

    x = database['Data_test_2']
    y = database['label_test_2']

    return x, y

## Import train data
data = pd.read_csv("../data/data/node04/data.csv", index_col=0)
x_train = data[data.columns[0:512]]
y_train = data[data.columns[-1]]

## Import test data
x_test, y_test = getTestDataset()
y_test = np.reshape(y_test, (y_test.shape[0],))

## Scale train and test data
scaler = MinMaxScaler()
scaler_model = scaler.fit(x_test)
x_train_scaled = scaler_model.transform(x_train)
x_test_scaled = scaler_model.transform(x_test)

## Creating Model
model = tf.keras.models.Sequential([
    layers.Dense(32, activation='relu', input_shape=(512,)),
    layers.Dense(16),
    layers.Dense(8, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

accuracy_list = []
loss_list = []

## Flower Client
class CifarClient(fl.client.NumPyClient):
    def __init__(self):
        self.totalRowCount = data.shape[0]
        self.currentIndex = 0
        self.continousTrainingBatchSize = 60
        self.nextIndex = self.currentIndex + self.continousTrainingBatchSize

    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train_scaled[self.currentIndex:self.nextIndex], y_train[self.currentIndex:self.nextIndex], epochs=1)
        self.currentIndex = self.nextIndex
        self.nextIndex = self.currentIndex + self.continousTrainingBatchSize if self.currentIndex + self.continousTrainingBatchSize < self.totalRowCount else self.totalRowCount
        return model.get_weights(), len(x_train_scaled), {}
    
    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test_scaled, y_test)
        loss_list.append(loss)
        accuracy_list.append(accuracy)
        return loss, len(x_test_scaled), {"Accuracy": accuracy}

## Start Client
server_address='localhost:'+str(sys.argv[1])
fl.client.start_numpy_client(server_address, client=CifarClient())

## Writing results to file
accuracy_list = np.array(accuracy_list)
loss_list = np.array(loss_list)
np.savetxt('./results/accuracy_client_4.txt', accuracy_list)
np.savetxt('./results/loss_client_4.txt', loss_list)