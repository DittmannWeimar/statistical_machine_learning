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
    result = [[]] * len(labels)
    for i, label in enumerate(labels):
        vector = [0] * len(numbers)
        for j in range(0, len(vector)):
            if (label == j):
                vector[j] = 1
            else:
                vector[j] = 0
        result[i] = vector
    return result

data = ciphers_pca
class_matrix = to_class_matrix(data[:,1].astype(int))
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

n_features = len(data.T)-2
n_outputs = len(class_matrix[0])

model = models.Sequential(name="DeepNN", layers=[
    ### hidden layer 1
    layers.Dense(name="h1", input_dim=n_features,
                 units=int(round((n_features+1)/2)), 
                 activation='sigmoid'),
    
    ### hidden layer 2
    layers.Dense(name="h2", units=int(round((n_features+1)/4)), 
                 activation='sigmoid'),
    
    ### layer output
    layers.Dense(name="output", units=n_outputs, activation='sigmoid')
])
model.summary()

model.compile(optimizer='adam', loss=loss_fn, 
              metrics=['accuracy',F1])

print("training..")
training = model.fit(x=data[:, 2:], y=class_matrix, batch_size=32, epochs=5, validation_split=0.3)

# plot
metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]    
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15,3))

## training    
ax[0].set(title="Training")    
ax11 = ax[0].twinx()    
ax[0].plot(training.history['loss'], color='black') 
ax[0].set_xlabel('Epochs')    
ax[0].set_ylabel('Loss', color='black')    
for metric in metrics:        
    ax11.plot(training.history[metric], label=metric)    
ax11.set_ylabel("Score", color='steelblue')    
ax11.legend()
        
## validation    
ax[1].set(title="Validation")    
ax22 = ax[1].twinx()    
ax[1].plot(training.history['val_loss'], color='black')    
ax[1].set_xlabel('Epochs')    
ax[1].set_ylabel('Loss', color='black')    
for metric in metrics:          
    ax22.plot(training.history['val_'+metric], label=metric)   
ax22.set_ylabel("Score", color="steelblue")

plt.savefig("plots/ann/training.png")
plt.show()