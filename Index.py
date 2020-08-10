#!/usr/bin/env python
# coding: utf-8

# # Welcome to Jupyter!

# In[1]:


#pip install quandl


# This repo contains an introduction to [Jupyter](https://jupyter.org) and [IPython](https://ipython.org).
# 
# Outline of some basics:
# 
# * [Notebook Basics](../examples/Notebook/Notebook%20Basics.ipynb)
# * [IPython - beyond plain python](../examples/IPython%20Kernel/Beyond%20Plain%20Python.ipynb)
# * [Markdown Cells](../examples/Notebook/Working%20With%20Markdown%20Cells.ipynb)
# * [Rich Display System](../examples/IPython%20Kernel/Rich%20Output.ipynb)
# * [Custom Display logic](../examples/IPython%20Kernel/Custom%20Display%20Logic.ipynb)
# * [Running a Secure Public Notebook Server](../examples/Notebook/Running%20the%20Notebook%20Server.ipynb#Securing-the-notebook-server)
# * [How Jupyter works](../examples/Notebook/Multiple%20Languages%2C%20Frontends.ipynb) to run code in different languages.

# In[2]:


import quandl
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split


# You can also get this tutorial and run it on your laptop:
# 
#     git clone https://github.com/ipython/ipython-in-depth
# 
# Install IPython and Jupyter:
# 
# with [conda](https://www.anaconda.com/download):
# 
#     conda install ipython jupyter
# 
# with pip:
# 
#     # first, always upgrade pip!
#     pip install --upgrade pip
#     pip install --upgrade ipython jupyter
# 
# Start the notebook in the tutorial directory:
# 
#     cd ipython-in-depth
#     jupyter notebook

# In[3]:


df = quandl.get("WIKI/AMZN")
print(df.head())


# In[4]:


df = df[['Adj. Close']] 
print(df.head())


# In[8]:


forecast_out=int(input("Enter number of further days to predict"))
print(forecast_out)


# In[9]:


df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)
print(df.tail())


# In[10]:


X = np.array(df.drop(['Prediction'],1))
X = X[:-forecast_out]
print(X)


# In[11]:


y = np.array(df['Prediction'])
y = y[:-forecast_out]
print(y)


# In[12]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[13]:


svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1) 
svr_rbf.fit(x_train, y_train)


# In[14]:


svm_confidence = svr_rbf.score(x_test, y_test)
print("svm confidence: ", svm_confidence)


# In[15]:


lr = LinearRegression()
lr.fit(x_train, y_train)


# In[16]:


lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ", lr_confidence)


# In[17]:


x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
print(x_forecast)


# In[18]:


lr_prediction = lr.predict(x_forecast)
print(lr_prediction)


# In[19]:


svm_prediction = svr_rbf.predict(x_forecast)
print(svm_prediction)


# In[ ]:




