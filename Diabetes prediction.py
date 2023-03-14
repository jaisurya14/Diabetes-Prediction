#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv(r'C:\Users\ganes\Downloads\diabetes.csv')


# In[3]:


data.isna().sum()


# In[4]:


data.head()


# In[5]:


data.shape


# In[6]:


data.info


# In[58]:


test_data = (8,183,64,0,0,23.3,.672,32)
test_data_np = np.asarray(test_data)


# In[7]:


data.columns


# In[8]:


data.describe


# In[11]:


data['Outcome'].value_counts(normalize=True)


# In[12]:


sns.countplot(x='Outcome',data=data)


# In[14]:


plt.figure(2)
plt.subplot(121)
sns.distplot(data['Pregnancies'])
plt.subplot(122)
data['Pregnancies'].plot.box(figsize=(15,5))


# In[15]:


plt.figure(2)
plt.subplot(121)
sns.distplot(data['Glucose'])
plt.subplot(122)
data['Glucose'].plot.box(figsize=(15,5))


# In[16]:


plt.figure(2)
plt.subplot(121)
sns.distplot(data['BloodPressure'])
plt.subplot(122)
data['BloodPressure'].plot.box(figsize=(15,5))


# In[17]:


sns.pairplot(data)


# In[18]:


data.corr()


# In[19]:


corr_matrix = data.corr()
sns.heatmap(corr_matrix,cmap='coolwarm')


# In[20]:


ax = plt.subplots(figsize=(12,8))
sns.heatmap(corr_matrix,cmap='coolwarm')


# In[21]:


data.groupby('Outcome').mean()


# In[22]:


X = data.drop(columns='Outcome', axis=1)
Y= data['Outcome']


# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=.2,random_state=123)


# In[25]:


from sklearn.linear_model import LogisticRegression


# In[26]:


logreg = LogisticRegression()


# In[27]:


logreg.fit(X_train,Y_train)


# In[28]:


logreg_predict = logreg.predict(X_test)


# In[29]:


from sklearn.metrics import accuracy_score 


# In[30]:


accuracy_score(Y_test,logreg_predict)


# In[59]:


test_data_rs = test_data_np.reshape(1,-1)


# In[63]:


test_predict = logreg.predict(test_data_rs)


# In[64]:


test_predict


# In[31]:


from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:





# In[32]:


print(confusion_matrix(Y_test, logreg_predict))


# In[33]:


print(classification_report(Y_test, logreg_predict))


# In[34]:


from sklearn import svm


# In[35]:


svm_model = svm.SVC(kernel='linear')


# In[36]:


from sklearn.ensemble import RandomForestClassifier 


# In[37]:


rfc_model = RandomForestClassifier(n_estimators=200)


# In[38]:


rfc_model.fit(X_train,Y_train)


# In[39]:


rfc_predict = rfc_model.predict(X_test)


# In[40]:


accuracy_score(Y_test, rfc_predict)


# In[42]:


rfc_model.feature_importances_


# In[48]:


pd.Series(rfc_model.feature_importances_, index = X.columns).plot(kind='barh')


# In[52]:


from sklearn.tree import DecisionTreeClassifier


# In[53]:


from xgboost import XGBClassifier


# In[54]:


pip install xgboost


# In[55]:


from sklearn.tree import DecisionTreeClassifier


# In[56]:


from xgboost import XGBClassifier


# In[ ]:




