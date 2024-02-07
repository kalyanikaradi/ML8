#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor#regression checking
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from  sklearn.metrics import accuracy_score, confusion_matrix,classification_report,recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[5]:


data=pd.read_csv('diabetes1.csv')


# In[6]:


data.head()


# In[7]:


data.tail()


# In[8]:


data.isnull().sum()


# In[11]:


data.info()


# In[12]:


data.describe()


# In[10]:


#box plot
plt.figure(figsize=(20,25), facecolor='white')
plotnumber=1

for column in data:
        if plotnumber<=9:
            ax=plt.subplot(3,3,plotnumber)
            sns.distplot(data[column])
            plt.xlabel(column,fontsize=20)
        plotnumber+=1
plt.show()


# In[14]:


#replace zero value with mean of column
data['BMI']=data['BMI'].replace(0,data['BMI'].median())
data['BloodPressure']=data['BloodPressure'].replace(0,data['BloodPressure'].median())
data['Insulin']=data['Insulin'].replace(0,data['Insulin'].median())
data['SkinThickness']=data['SkinThickness'].replace(0,data['SkinThickness'].median())


# In[15]:


#lets see how data is distributes
plt.figure(figsize=(20,25), facecolor='white')
plotnumber=1

for column in data:
        if plotnumber<=9:
            ax=plt.subplot(3,3,plotnumber)
            sns.distplot(data[column])
            plt.xlabel(column,fontsize=20)
        plotnumber+=1
plt.show()


# In[17]:


#model Creation
#split x and y
X=data.drop(columns=['Outcome'])
Y=data['Outcome']


# In[18]:


X


# In[19]:


Y


# In[38]:


scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)


# In[39]:


X_scaled
#splitting the training and testing


# In[50]:


X_train,X_test,Y_train,Y_test=train_test_split(X_scaled,Y,random_state=42)


# In[51]:


#creating list to store error value
error_rate=[]
for i in range(1,11):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train,Y_train)
        pred_i=knn.predict(X_test)
        error_rate.append(np.mean(pred_i != Y_test))#i is predict value and y is tested value


# In[52]:


error_rate


# In[53]:


#lets plot the k value and error rate
plt.figure(figsize=(10,6))
plt.plot(range(1,11),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error Rate vs. K value')
plt.xlabel('K')
plt.ylabel('Error rate')
plt.show()


# In[54]:


knn1 = KNeighborsClassifier(n_neighbors=5)
knn1.fit(X_train,Y_train)


# In[55]:


#predict
Y_pred = knn1.predict(X_test)


# In[56]:


#checking accuracy score
print('The accuracy score is :', accuracy_score(Y_test,Y_pred))


# In[57]:


print(classification_report(Y_test,Y_pred))


# In[58]:


recall=recall_score(Y_test,Y_pred)
recall


# In[59]:


#checking balance target
sns.catplot(x='Outcome',data=data,kind='count')


# In[60]:


data.Outcome.value_counts()


# In[ ]:




