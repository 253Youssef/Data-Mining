#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


# # Functions

# In[37]:


def pca(df):
    
    # Get values of dataframe
    X = df.values

    # Standardise the data values
    X_std = StandardScaler().fit_transform(X)
    
    # Get the mean vectore of the data
    mean_vec = np.mean(X_std, axis=0)

    # Subtract mean from data
    X_std_mean = (X_std - mean_vec)

    # Get transpose of data to multiply it by the untransposed data to get the covariance matrix
    X_std_mean_transpose = X_std_mean.T

    # Calculate the covariance matrix
    cov_mat = X_std_mean_transpose.dot(X_std_mean) / (X_std.shape[0]-1)
    
    # Compute the eigen values and vectors
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    
    # Compute the projection matrix based on the eigen vectors
    num_features = X.shape[1]
    proj_mat = eig_pairs[0][1].reshape(num_features,1)
    for eig_vec_idx in range(1, X_std.shape[1]):
        proj_mat = np.hstack((proj_mat, eig_pairs[eig_vec_idx][1].reshape(num_features,1)))

    # Project the data 
    pca_data = X_std.dot(proj_mat)
    
    # Return projection matrix and the pca data
    return proj_mat, pca_data, eig_vecs

def scatterFigure(pca_data, file_name):
    
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 10)
    ax.grid()
    
    principal_component_1 = pca_data.T[0]
    principal_component_2 = pca_data.T[1]

    # Plot the 1st principal component on the x-axis with the 2nd component on the y-axis
    plt.scatter(principal_component_1, principal_component_2, color='blue', marker='o')
    
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title('Principal Components')

    plt.title('Principal Components')
    
    plt.savefig(file_name + '.png', dpi = 300)
    
def scatterFigureAxes(data, file_name, eig_vecs):
    
    # Get X and Y values of Component Axes
    x_v1, y_v1 = eig_vecs[0][0], eig_vecs[0][1]  
    x_v2, y_v2 = eig_vecs[1][0], eig_vecs[1][1]
        
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 10)
    ax.grid()
    
    plt.plot([x_v2*-5, x_v2*5],
             [y_v2*-5, y_v2*5], color='red', label = 'Principal Component 1')
    
    plt.plot([x_v1*-2, x_v1*2],
             [y_v1*-2, y_v1*2], color='green', label = 'Principal Component 2')


    # Plot the 1st principal component on the x-axis with the 2nd component on the y-axis
    plt.scatter(data.T[0], data.T[1], color='blue', marker='o')
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Principal Components')
    ax.legend()

    plt.title('Principal Components Axes on Original Data')
    
    
    plt.axis('equal')
    
    
    plt.savefig(file_name + '.png', dpi = 300)
    
def histogram(home, away, file_name):
    
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 10)
    ax.grid()
    
    plt.hist([home, away], bins = 10, label = ['Home', 'Away'], color = ['r', 'b'])
    plt.legend(loc='upper right')
    
    plt.title(file_name)

    plt.savefig(file_name + '.png', dpi = 300)
    
def plotDistance(principal_components_axis, difference_array, file_name):
    
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 10)
    ax.grid()

    # Plot the 1st principal component on the x-axis with the 2nd component on the y-axis
    plt.plot(principal_components_axis, difference_array, color='blue', marker='o')
    
    ax.set_xlabel('Principal Components')
    ax.set_ylabel('Difference')
    ax.set_title('Principal Components vs. Differences')

    plt.title('Principal Components vs. Differences')
    
    plt.savefig(file_name + '.png', dpi = 300)


# # Problem 1

# ## Read Data Text File

# In[3]:


df1 = pd.read_csv('Data.txt', sep=" ", header = None, dtype = np.float64)


# ## Get Dataframe Info

# In[4]:


# Get shape of dataframe
df1.shape


# In[5]:


# Get head of the dataframe
df1.head()


# In[6]:


# Describe dataframe with meaningful insights such as mean and standard deviation
df1.describe()


# ## Apply PCA to the Data

# In[7]:


# Get projection matrix and projected data from pca function
proj_mat1, pca_data1, eig_vecs1 = pca(df1)


# ## Plots

# ### Principal Components

# In[28]:


# Plot and save figure showing projected data on the 2 components
scatterFigure(pca_data1, 'Principal_Components')


# ### Original Data with Principal Component Axes

# In[29]:


# Get values of dataframe
X1 = df1.values

# Standardise the data values
X_std1 = StandardScaler().fit_transform(X1)

# Plot original data with axes shown on the figure, and save the figure as 'Data_PCA'
scatterFigureAxes(X_std1, 'Data_PCA', eig_vecs1)


# # Problem 2

# ## Read EPL Excel File

# In[10]:


df2 = pd.read_excel('EPL.xlsx')


# ## Get Dataframe Info

# In[11]:


# Get shape of dataframe
df2.shape


# In[12]:


# Get head of the dataframe
df2.head()


# In[13]:


# Describe dataframe with meaningful insights such as mean and standard deviation
df2.describe()


# ## Get Indices for Home and Away Team Wins

# In[14]:


# Get indices for datapoints where Home Team won
home_indices = list(df2.index[df2['FTR'] == 'H'])

# Get indices for datapoints where Away Team won
away_indices = list(df2.index[df2['FTR'] == 'A'])


# ## Apply PCA to the Data

# In[15]:


# Get relevant columns to compute PCA
columns = ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC']
proj_mat2, pca_data2, eig_vecs2 = pca(df2[columns])


# ## Split Projected Data into Home and Away Values

# In[16]:


# Get projected home data
pca_data2_home = [ pca_data2[index] for index in home_indices ]

# Get Away Projected Data
pca_data2_away = [ pca_data2[index] for index in away_indices ]


# In[17]:


# Get home projected for each principal component
projected_data_home = list(np.array(pca_data2_home).T)

# Get away projected for each principal component
projected_data_away = list(np.array(pca_data2_away).T)

# Zip both projected data for each principal component
projected_data = list(zip(projected_data_home, projected_data_away))


# ## Plot Home and Away Projections for each Principal Component

# In[22]:


# Create index for each principal component
pca_index = 1

# Loop over projections for each principal component of home and away data
for projected_home, projected_away in projected_data:
        
    # Plot and save histogram of the principal component
    histogram(projected_home, projected_away, 'Proj_PC' + str(pca_index))

    # Increment the principal component index
    pca_index = pca_index + 1


# ## Distance Comparison and Plot

# In[39]:


# Define x-axis array for principal components
principal_components_axis = range(1, 9)

# Define array to hold differences between home and away projections' means
difference_array = []

# Loop over projections for each principal component of home and away data
for projected_home, projected_away in projected_data:
        
    # Get the mean value of the home projections for this principall component
    home_mean = projected_home.mean()
    
    # Get the mean value of the away projections for this principall component
    away_mean = projected_away.mean()
    
    # Compute difference between home and away means
    difference = home_mean - away_mean
    
    # Append difference to differences array
    difference_array.append(difference)
    
# Plot and save components vs. distances figure
plotDistance(principal_components_axis, difference_array, 'Distance')


# In[ ]:




