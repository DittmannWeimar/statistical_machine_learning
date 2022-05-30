from sklearn.model_selection import cross_validate
import tensorflow as tf
from tensorflow.keras import models, layers, utils, backend as K
import matplotlib.pyplot as plt
import shap
import numpy as np
import random
import statistics
from stopwatch import Stopwatch
import matplotlib.pyplot as plt
import seaborn as sn
import math
import os
import pandas as pd
import timeit
import rpy2.robjects as ro
from rpy2.robjects import r
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import requests
from sklearn.decomposition import PCA
from threading import Thread
import json

data_url = "https://nextcloud.sdu.dk/index.php/s/Zzjcqjopy5cTawn/download/data_33.Rdata"
data_file = "data.Rdata"

#download the file
req = requests.get(data_url, allow_redirects=True, stream=True)

#save downloaded file
with open(data_file,"wb") as rf:
     for chunk in req.iter_content(chunk_size=1024):
         # writing one chunk at a time to r file
         if chunk:
              rf.write(chunk)

r_data=r.load(data_file)

with localconverter(ro.default_converter + pandas2ri.converter):
    ciphers = ro.conversion.rpy2py(r['ciphers'])

def take(arr, indexes):
    return arr[indexes]

def take_inverse(arr, indexes):
    mask = np.ones(len(arr), dtype=bool)
    mask[indexes] = False
    return arr[mask]

def all_persons_in(ciphers, split = 0.5):
    count = len(ciphers)
    samples = random.Random(666).sample(range(0, count), int(count * split))
    train_data = take(ciphers, samples)
    test_data = take_inverse(ciphers, samples)
    return { "train_data": train_data, "test_data": test_data }

def generate_pca(ciphers, cumsum):
    data = ciphers[:, 2:]

    pca = PCA(cumsum)
    pca.fit(data)

    return pca

def decompose(ciphers, cumsum):
    cipher_length = len(ciphers.T)

    metadata = np.delete(ciphers, range(2, cipher_length), axis=1)
    data = ciphers[:, 2:]

    pca = generate_pca(ciphers, cumsum)
    transformed = pca.transform(data)
    ciphers_pca = np.append(metadata, transformed, axis=1)
    return ciphers_pca

ciphers_pca = decompose(ciphers, 0.8)

# define metrics
def Recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def Precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def F1(y_true, y_pred):
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def to_class_matrix(labels):
    numbers = set(labels)
    result = [[0] * len(numbers)] * len(labels)
    for i, label in enumerate(labels):
        vector = result[i]
        for j in range(0, len(vector)):
            if (label == j):
                vector[j] = 1.0
            else:
                vector[j] = 0.0
        result[i] = vector
    return np.array(result)

def split(ciphers):
    train_data = ciphers[:, 2:]
    train_labels = ciphers[:, 1]
    train_person = ciphers[:, 0]
    data = []
    for cipher in ciphers:
        data.append(cipher[2:len(cipher)])
    return { "person": train_person, "truth":train_labels, "data": train_data}

def highest_probability_to_class(predictions):
    result = [-1] * len(predictions)
    for i, prediction in enumerate(predictions):
        max_value = max(prediction)
        max_index = np.where(prediction == max_value)[0][0]
        result[i] = max_index
    return result

def compute_accuracy_folds(folds, ciphers, predictor, **kwargs):
    sw = Stopwatch(True)
    results = []
    for fold in folds:
        train_raw = split(take_inverse(ciphers, fold))
        test_raw = split(take(ciphers, fold))

        train_data = train_raw['data']
        train_labels = train_raw['truth']
        test_data = test_raw['data']
        test_labels = test_raw['truth']

        predictions = predictor(train_data, train_labels, test_data, **kwargs)
        cp = highest_probability_to_class(predictions)

        cf = pd.crosstab(cp, test_labels)
        accuracy = np.diag(cf).sum() / cf.to_numpy().sum()

        results.append(accuracy)

    mean = statistics.mean(results)
    stdev = statistics.stdev(results)
    time = sw.end()
    return { "mean": mean, "stdev": stdev, "time": time }

def k_folds(arr_size, folds):
    samples = range(0, arr_size)
    return np.array_split(samples, folds)

data = ciphers_pca
class_matrix = to_class_matrix(data[:,1].astype(int))
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def generate_model(conv_size, num_outputs, num_filters, activation_function):

    network_layers = []
    network_layers.append(layers.InputLayer(input_shape=(num_inputs,)))


    network_layers.append(layers.Conv2D())

    for i in range(+)

    for i in range(0, num_dense_layers):
        nodes = int(np.interp(i, [0, num_dense_layers], [num_inputs, num_outputs]))
        network_layers.append(layers.Dense(
                name = "h{0}".format(i),
                units=nodes,
                activation=activation_function
            ))

    network_layers.append(layers.Dense(name="output", units=num_outputs, activation=activation_function))

    model = models.Sequential(name="CipherPerceptron", layers=network_layers)
    return model

def train_network (train_data, train_labels, activation_function, dense_layers, save_as=None, **kwargs):

    n_features = len(train_data.T)
    n_outputs = len(set(train_labels))

    # Create model
    print("Generating model with {0} features, {1} outputs, {2} hidden layers, and activation function '{3}'..".format(n_features, n_outputs, dense_layers, activation_function))
    model = generate_model(n_features, n_outputs, dense_layers, activation_function)
    model.summary()

    # Compile
    print("Compiling model with {0} features, {1} outputs, {2} hidden layers, and activation function '{3}'..".format(n_features, n_outputs, dense_layers, activation_function))
    model.compile(optimizer='adam', loss=loss_fn, 
        metrics=['accuracy',F1])

    # Train
    print("Training model with {0} features, {1} outputs, {2} hidden layers, and activation function '{3}'..".format(n_features, n_outputs, dense_layers, activation_function))
    sw = Stopwatch(True)
    training = model.fit(x=train_data, y=train_labels, **kwargs)
    if (save_as is not None):
        model.save(save_as)
    time = sw.end()
    return (model, training, time)

def plot_training (training, save_as=None, show=False):
    # plot
    metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15,3))

    ## training    
    ax[0].set(title="Training")    
    ax11 = ax[0].twinx()    
    ax[0].plot(training.history['loss'], color='black', label="loss") 
    ax[0].set_xlabel('Epochs')    
    ax[0].set_ylabel('Loss', color='black')    
    for metric in metrics:        
        ax11.plot(training.history[metric], label=metric)    
    ax11.set_ylabel("Score", color='steelblue')    
    ax11.legend()
                    
    ## validation    
    ax[1].set(title="Validation")    
    ax22 = ax[1].twinx()    
    ax[1].plot(training.history['val_loss'], color='black', label="loss")    
    ax[1].set_xlabel('Epochs')    
    ax[1].set_ylabel('Loss', color='black')    
    for metric in metrics:          
        ax22.plot(training.history['val_'+metric], label=metric)   
    ax22.set_ylabel("Score", color="steelblue") 

    if (save_as is not None):
        plt.savefig(save_as)
    if (show):
        plt.show()

def ann_predictor(train_data, train_labels, test_data, **kwargs):
    activation_function = kwargs["activation_function"]
    hidden_layers = kwargs["hidden_layers"]
    del kwargs["activation_function"]
    del kwargs["hidden_layers"]
    model, training, time = train_network(train_data, train_labels, activation_function, hidden_layers, **kwargs)
    predictions = model.predict(test_data)
    return predictions

#results["sigmoid_api_pca"] = compute_accuracy_folds(k_folds(size, 11), ciphers_shuffle_pca, ann_predictor, activation_function="sigmoid", hidden_layers=10, batch_size=32, epochs=200, shuffle=False, verbose=0, validation_split=0.3)