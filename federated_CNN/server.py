## Import Modules
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import flwr as fl
from sklearn.preprocessing import MinMaxScaler
import scipy.io as sio
import sys

## Method to get validation data
def getTestDataset():
    database = sio.loadmat('../testData/data_base_all_sequences_random.mat')

    x = database['Data_test_2']
    y = database['label_test_2']

    return x, y

## Import validation data
x_val, y_val = getTestDataset()
y_val = np.reshape(y_val, (y_val.shape[0],))

## Scale validation data
scaler = MinMaxScaler()
scaler_model = scaler.fit(x_val)
x_val = scaler_model.transform(x_val)


## Load and compile model for server-side parameter evaluation
model = tf.keras.models.Sequential([
    layers.Reshape((512,1), input_shape=(512,1)),
    layers.Conv1D(16, 3, activation='relu'),
    layers.MaxPooling1D(3),
    layers.Conv1D(4, 3, activation='relu'),
    layers.Flatten(),
    layers.Dense(8, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.build(input_shape=(512, 1))

accuracy_list = []
loss_list = []

# Centralized Evaluation for Server's Strategy
def get_eval_fn(model):
    '''Return an evaluation function for server side evaluation'''

    def evaluate(weights: fl.common.Weights):
        model.set_weights(weights)
        loss, accuracy = model.evaluate(x_val, y_val)
        loss_list.append(loss)
        accuracy_list.append(accuracy)
        return loss, {"Accuracy": accuracy}

    return evaluate     

strategy = fl.server.strategy.FedAvg( 
    min_fit_clients = 5,
    min_eval_clients = 5,
    min_available_clients = 5,  
    eval_fn = get_eval_fn(model)
)

# Start Server
fl.server.start_server(
    strategy = strategy,
    server_address='localhost:'+str(sys.argv[1]),
    config={"num_rounds": 54}
)

## Writing results to file
accuracy_list = np.array(accuracy_list)
loss_list = np.array(loss_list)
np.savetxt('./results/accuracy_server.txt', accuracy_list)
np.savetxt('./results/loss_server.txt', loss_list)