#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_excel('REGRESSIONanals.xlsx')


# In[3]:


df


# In[4]:


(df.info())


# In[5]:


(df['Engagement'].value_counts())


# In[6]:


(df.describe())


# In[7]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[8]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# In[9]:


print(df.isnull().sum())


# In[10]:


df.dropna(inplace=True)


# In[11]:


df


# In[12]:


X = df[['Trust', 'Engagement', 'PERCEPTION', 'GENZ_MARKETING']]
y = df['Purchase']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[14]:


model = LogisticRegression()
model.fit(X_train, y_train)


# In[15]:


y_pred = model.predict(X_test)


# In[17]:


coefficients = model.coef_
intercept = model.intercept_


# In[18]:


log(odds of Purchase) = intercept + coef1 * Trust + coef2 * Engagement + coef3 * PERCEPTION + coef4 * GENZ_MARKETING


# In[19]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Assuming your data is stored in a DataFrame named 'df'
X = df[['Trust', 'Engagement', 'PERCEPTION', 'GENZ_MARKETING']]
y = df['Purchase']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(confusion)

report = classification_report(y_test, y_pred)
print('Classification Report:')
print(report)


# In[20]:


# Assuming your logistic regression model is named 'model'
coefficients = model.coef_
intercept = model.intercept_

print('Coefficients (weights):')
print(coefficients)

print('\nIntercept:')
print(intercept)


# In[22]:


import statsmodels.api as sm

X = df[['Trust', 'Engagement', 'PERCEPTION', 'GENZ_MARKETING']]
y = df['Purchase']

# Add a constant term to the independent variables (intercept)
X = sm.add_constant(X)

# Fit the logistic regression model
model = sm.Logit(y, X).fit()

# Get the summary of the model
summary = model.summary()

# Display the summary
print(summary)


# In[23]:


print('summary') 


# In[24]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Assuming your data is stored in a DataFrame named 'df'
X = df[['Trust', 'Engagement', 'PERCEPTION', 'GENZ_MARKETING']]
y = df['Purchase']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier
model = DecisionTreeClassifier(random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(confusion)

report = classification_report(y_test, y_pred)
print('Classification Report:')
print(report)


# In[ ]:




