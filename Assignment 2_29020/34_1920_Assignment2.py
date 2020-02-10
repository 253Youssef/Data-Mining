#!/usr/bin/env python
# coding: utf-8

# # Team Info

# Member Name: Youssef Ayman Taher Kandil
# 
# Member ID: 34-1920

# # Libraries

# In[13]:


import pandas as pd
import numpy as np
import cv2

import os as os
from os import listdir
from os.path import isfile, join

import scipy.stats

import matplotlib.pyplot as plt


# # Define Directories

# In[14]:


train_directory = 'Train/'
test_directory = 'Test/'

train_files = [f for f in listdir(train_directory) if isfile(join(train_directory, f))]
test_files = [f for f in listdir(test_directory) if isfile(join(test_directory, f))]


# # Creating Dataframes

# ## Train Dataframe

# In[15]:


# Read only the 1st image to get the dimensions of the images and create pandas dataframe
gray = cv2.imread(train_directory + '/' + train_files[0], 0)
num_columns = range(len(gray.flatten()))
columns = list(num_columns) + ['class']
columns = [str(feature) for feature in columns]

# Create rows array to append image flattened array to
train_rows = []

for file in train_files:
    gray = cv2.imread(train_directory + '/' + file, 0)
    c = file[2]
    gray_flattened = gray.flatten() / 255
    row = list(gray_flattened) + [c]
    train_rows.append(row)
    
train_df = pd.DataFrame(train_rows, columns=columns)
train_df.to_csv('train_df.csv', index = False)


# ## Test Dataframe

# In[16]:


# Read only the 1st image to get the dimensions of the images and create pandas dataframe
gray = cv2.imread(test_directory + '/' + test_files[0], 0)
num_columns = range(len(gray.flatten()))
columns = list(num_columns) + ['class']
columns = [str(feature) for feature in columns]

test_rows = []

for file in test_files:
    gray = cv2.imread(test_directory + '/' + file, 0)
    c = file[2]
    gray_flattened = gray.flatten() / 255
    row = list(gray_flattened) + [c]
    test_rows.append(row)
    
test_df = pd.DataFrame(test_rows, columns=columns)
test_df.to_csv('test_df.csv', index = False)


# # "Training"

# In[17]:


train_dict = {}

unique_classes = list(set((train_df['class']).tolist()))

for c in unique_classes:
    train_dict[c] = {}
    class_df = train_df.loc[train_df['class'] == c]
    mean = np.mean(class_df)
    std = np.std(class_df)
    train_dict[c]['mean'] = mean
    train_dict[c]['std'] = std
    train_dict[c]['probability'] = len(class_df) / len(train_df)
    train_dict[c]['gaus'] = scipy.stats.norm(mean, std)


# # Create Right  & Wrong Predictions Dictionaries

# In[18]:


correct_predictions = {}
for c in unique_classes:
    correct_predictions[c] = 0
    
wrong_predictions = {}
for c in unique_classes:
    wrong_predictions[c] = {}
    for c2 in set((train_df['class']).tolist()):
        if (c!=c2):
            wrong_predictions[c][c2] = 0


# # Predictions

# In[19]:


for index, datapoint in test_df.iterrows():
        
    actual_class = datapoint['class']
    
    datapoint_features = pd.to_numeric(datapoint[list(num_columns)], errors='coerce')

    pred_dict = {}
    
    for c in unique_classes:
        
        gaus = train_dict[c]['gaus']
        class_probability = train_dict[c]['probability']
        
        probabilities = gaus.pdf(datapoint_features)
        probabilities[(np.isnan(probabilities)) | (probabilities < 0.1)] = 0.1
        product_probability = np.prod(probabilities)
        
        pred_probability = product_probability * class_probability
        
        pred_dict[c] = pred_probability
        
    max_class = max(pred_dict, key=lambda key: pred_dict[key])
    
    if actual_class == max_class:
        correct_predictions[actual_class] += 1 
    else:
        wrong_predictions[actual_class][max_class] += 1


# # Write Predictions to Files

# In[20]:


predictions_directory = 'Predictions/'

if not os.path.exists(predictions_directory):
            os.makedirs(predictions_directory)

output = open("correct_predictions.txt", "w+")
correct_predictions_sorted = sorted(correct_predictions.items(), key=lambda kv: -1*kv[1])

output_string = ''
for key in correct_predictions_sorted:
    output_string = output_string + (key[0] + ': ' + str(key[1]) + '\n')
output.write(output_string)

for c in wrong_predictions:
    output = open(predictions_directory + c + ".txt", "w+")
    output_string = ''
    for c2 in wrong_predictions[c]:
        output_string = output_string + (c2 + ': ' + str(wrong_predictions[c][c2]) + '\n')
    output.write(output_string)


# # Bar Chart for Correct Predictions

# In[21]:


plt.bar(range(len(correct_predictions)), list(correct_predictions.values()))
plt.xticks(range(len(correct_predictions)), list(correct_predictions.keys()))
plt.savefig('Accuracy.jpg', dpi=100)
plt.show


# In[ ]:




