#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


lgr=pd.read_csv("C:/Users/Admin/Desktop/pima-indians-diabetes .csv")


# In[3]:


lgr.head()


# In[4]:


lgr.isna().sum()


# In[5]:


lgr.describe()

lgr.info()
# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[8]:


X = lgr.drop('class', axis=1)
y = lgr['class']


# In[9]:


X


# In[10]:


y


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[12]:


X_train


# In[13]:


y_train


# In[14]:


model = LogisticRegression()


# In[15]:


model


# In[16]:


model.fit(X_train, y_train)


# In[17]:


y_pred = model.predict(X_test)


# In[18]:


y_pred


# In[19]:


accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{classification_report_result}')
print(f'Confusion Matrix:\n{conf_matrix}')


# In[20]:


sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-Diabetic', 'Diabetic'], 
            yticklabels=['Non-Diabetic', 'Diabetic'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[23]:


feature_of_interest = 'age'
sns.set(style="whitegrid")
sns.lmplot(x=feature_of_interest, y='class', data=lgr, logistic=True, height=6)
plt.title(f'Logistic Regression Plot for {feature_of_interest}')
plt.xlabel(feature_of_interest)
plt.ylabel('Probability of Being Diabetic')
plt.show()


# In[ ]:




