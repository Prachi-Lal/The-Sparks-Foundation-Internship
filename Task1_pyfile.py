#!/usr/bin/env python
# coding: utf-8

# ****PRACHI LAL****

# **Task1: Predicting the Score based on the number of hours studied**

# ***Importing Necessary Libraries***

# In[53]:


import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# ***Importing Data***

# In[3]:


df = pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
df


# ***Visualizing the Data***

# In[14]:


x = df['Hours']
y = df['Scores']
plt.scatter(x, y)
plt.xlabel('Hours of Study')
plt.ylabel('Score Achieved')
plt.show()


# ***Defining attributes and target for data splitting***

# In[17]:


x = df[['Hours']]
y = df['Scores']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 0)


# ***Model Development***

# In[18]:


reg = linear_model.LinearRegression()


# ***Model Training***

# In[19]:


reg.fit(xtrain,ytrain)


# ***Constructing the Regressor Line***

# In[23]:


line = reg.coef_*x + reg.intercept_
plt.scatter(x, y)
plt.plot(x, line);
plt.xlabel('Hours of Study')
plt.ylabel('Score Achieved')
plt.show()


# ***Actual Data v/s Predicted Data***

# In[44]:


ypred = reg.predict(xtest)
df = pd.DataFrame({'Actual': ytest, 'Predicted': ypred})  
df 


# ***Model Evaluation***

# ***Coefficient of determination or R squared Value***

# In[8]:


reg.score(x,y)


# ***Mean Squared Value***

# In[54]:


mean_absolute_error(ypred, ytest)


# ***Predicting Score for someone who studies for 9.25 hours***

# In[11]:


reg.predict(np.array([[9.25]]))

