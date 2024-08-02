#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install pandas


# In[3]:


pip install matplotlib


# In[4]:


pip install seaborn


# Import The Library
# 

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


# In[6]:


dataset = pd.read_csv('C:/Users/anujd/OneDrive/Desktop/50_Startups.csv')


# In[7]:


dataset.head()


# In[8]:


dataset.tail()


# In[9]:


dataset.describe()


# In[10]:


print('There are',dataset.shape[0],'rows and',dataset.shape[1],'colomns in the dataset')


# In[11]:


print('There are',dataset.duplicated().sum(),'duplicate values in data set')


# In[12]:


dataset.isnull().sum()


# In[13]:


dataset.info()


# In[14]:


c = dataset.corr()
c

    


# In[15]:


sns.heatmap(c,annot=True,cmap='Oranges')
plt.show()


# In[16]:


outliers=["Profit"]
plt.rcParams['figure.figsize']=[8,8]
sns.boxplot(data=dataset[outliers],orient='v',palette='Set2',width=0.7)
plt.title('Outliers Varriables Distribution')
plt.ylabel('Profit Range')
plt.xlabel('Continuos Varriable')
plt.show()


# In[17]:


sns.displot(dataset['Profit'],bins=10,kde=True,color='Blue')
plt.show()


# In[18]:


x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,:3].values


# In[19]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7,random_state=0)
x_train


# In[20]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
('Model has been trained Successfully')


# In[21]:


y_pred = model.predict(x_test)
y_pred


# In[22]:


testing_data_model_score = model.score(x_test,y_test)
testing_data_model_score


# In[23]:


df = pd.DataFrame(data={'Predicted Value':y_pred.flatten(),'Actual value':y_test.flatten()})
df


# In[24]:


from sklearn.metrics import r2_score
r2_score = r2_score(y_pred,y_test)
print('R2 Score of the model is',r2_score)


# In[25]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_pred,y_test)
print('Mean Squared error of the model is ', mse)


# In[26]:


import numpy as np
rmse = np.sqrt(mean_squared_error(y_pred,y_test))
print('Root mean Squared Error of the model is ',rmse)


# In[27]:


from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_pred,y_test)
print('Mean absolute error of the model is ',mae)


# In[ ]:





# In[ ]:





# In[ ]:




