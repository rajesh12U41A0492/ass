#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


nv=pd.read_csv("C:/Users/Admin/Downloads/xAPI-Edu-Data.csv")


# In[3]:


nv.head()


# In[4]:


nv.info()


# In[5]:


nv.describe()


# In[6]:


nv.isna().sum()


# In[17]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
categorical_columns = ['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'Class','GradeID', 'SectionID', 'Topic', 'Semester', 'Relation', 'ParentAnsweringSurvey', 'ParentschoolSatisfaction', 'StudentAbsenceDays']


# In[18]:


for column in categorical_columns:
    nv[column] = label_encoder.fit_transform(nv[column])


# In[19]:


X = nv.drop('Class', axis=1)
y = nv['Class']


# In[20]:


X


# In[21]:


y


# In[37]:


sns.heatmap(nv)


# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[24]:


X_train


# In[25]:


y_train


# In[26]:


model = GaussianNB()
model.fit(X_train, y_train)


# In[27]:


y_pred = model.predict(X_test)


# In[28]:


y_pred


# In[29]:


accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)


# In[30]:


accuracy


# In[31]:


classification_report_result


# In[32]:


conf_matrix


# In[34]:


plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(y_test.unique()),
            yticklabels=sorted(y_test.unique()))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[35]:


plt.figure(figsize=(8, 2))
sns.heatmap(pd.DataFrame.from_dict(classification_report(y_test, y_pred, output_dict=True)).iloc[:-1, :].T, annot=True, cmap='Blues')
plt.title('Classification Report')
plt.show()


# In[36]:


accuracy = accuracy_score(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.barplot(x=['Accuracy'], y=[accuracy])
plt.title('Model Accuracy')
plt.ylim(0, 1)
plt.show()


# In[ ]:




