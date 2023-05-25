#!/usr/bin/env python
# coding: utf-8

# <font color=red>Import Pandas and salary_dataset_1.csv file

# In[29]:


import pandas as pd
salary_data = pd.read_csv("salary_dataset_1.csv")
salary_data


# <font color=red>Revome Unnamed column
# <br><font color=black>Here Unnamed Column name was blanked in csv file

# In[30]:


salary_data = salary_data.drop(salary_data.columns[0], axis=1)
salary_data


# <font color=red>Find 'Number of rows' and 'Number of columns' in Dataset

# In[31]:


salary_data.shape


# <font color=red>Data Preparation and Train-Test Split for Machine Learning

# Data Splitting: Feature and Target Variable Separation
# <br>Drop 'Salary' Column and save as variable x and y
# <br><font color=green>Press only one time shift+Enter, function will run otherwise it will through error if we press more that one.

# In[32]:


x = salary_data.drop(['Salary'], axis=1)
y = salary_data['Salary']


# Splitting Data into Training and Testing Sets using scikit-learn's train_test_split
# <br><font color=green>Note: The test_size parameter is set to 0.2, which means that 20% of the data will be allocated to the testing set, while the remaining 80% will be used for training. 
# <br>The random_state parameter is set to 0 to ensure reproducibility of the split.

# In[33]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)


# Find 'Number of rows' and 'Number of columns' x_test

# In[35]:


x_test.shape


# Training a K-Nearest Neighbors Regressor Model

# In[37]:


from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()
knr.fit(x_train, y_train)


# Generating Predictions using the K-Nearest Neighbors Regressor Model

# In[39]:


y_pred = knr.predict(x_test)
y_pred


# Making Single Data Point Prediction using the K-Nearest Neighbors Regressor Model
# <br>Also import numpy

# In[40]:


import numpy as np
input_data = (2)  #(x), x is a random value
convert_to_array = np.asarray(input_data)
re_shape = convert_to_array.reshape(1,-1)
prediction = knr.predict(re_shape)
print(prediction)


# Assessing the Accuracy of the K-Nearest Neighbors Regressor Model with the score() Function

# In[41]:


knr.score(x_test, y_test)


# <font color=red>
# Loading and Exploring the Iris Dataset

# In[42]:


from sklearn.datasets import load_iris
iris = load_iris()
iris


# Creating a Pandas DataFrame from the Iris Dataset with Column Names

# In[43]:


dff = pd.DataFrame(iris.data,columns= iris.feature_names)
dff


# Target Class Names in the Iris Dataset

# In[44]:


iris.target_names


# In[ ]:




