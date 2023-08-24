#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[2]:


ab=pd.read_csv("C:\\Users\\Ganes\\OneDrive\\Documents\\titanic3.csv 1.csv")


# In[3]:


ab


# In[4]:


ab.info()


# In[5]:


ab.head(10)


# In[6]:


ab.tail(7)


# In[7]:


ab.sample(6)


# In[8]:


ab.describe


# In[9]:


ab.columns


# In[10]:


ab.isnull()


# In[11]:


ab.isnull().sum()


# In[12]:


ab.replace(' ',np.nan)


# In[13]:


ab.isnull().sum()*100/len(ab)


# In[14]:


ab.drop(['cabin','boat','body','home.dest'],axis=1,inplace=True)# Drop columns above 40% null values


# In[15]:


ab.info()


# In[16]:


ab.isnull().sum()*100/len(ab)


# * data[column_name].fillna(data[column_name].mode()[0])
# * data[column_name].fillna(data[column_name].mean())
# * data[column_name].fillna(data[column_name].median())
# 
# * null values greater than 40% --> the column will be dropped
# * null values will be less than 5%--> we can drop the null values
# * null values will be between 5 to 40-->#replace the values with either mean / median (depending on presence of outliers)
# 
# * data > object(string)---use mode
# 
# * data > numeric(numbers)-- use mean/median
# 
# * is we have outlier we use median otherwise use mean

# In[17]:


ab.info()


# In[18]:


ab['name'].fillna(ab['name'].mode()[0],inplace=True)


# In[19]:


ab['sex'].fillna(ab['sex'].mode()[0],inplace=True)
ab['ticket'].fillna(ab['ticket'].mode()[0],inplace=True)
ab['embarked'].fillna(ab['embarked'].mode()[0],inplace=True)


# In[20]:


ab.isnull().sum()*100/len(ab)


# In[21]:


ab.boxplot()


# In[22]:


ab['age'].fillna(ab['age'].median(),inplace=True)
ab['sibsp'].fillna(ab['sibsp'].median(),inplace=True)
ab['fare'].fillna(ab['fare'].median(),inplace=True)
ab['parch'].fillna(ab['parch'].median(),inplace=True)


# In[23]:


ab.info()


# In[24]:


ab.isnull().sum()


# In[25]:


ab['pclass'].fillna(ab['pclass'].mean(),inplace=True)
ab['survived'].fillna(ab['survived'].mean(),inplace=True)


# In[26]:


ab.isnull().sum()


# * There is no any missing value now all missing value is filled

# In[27]:


plt.plot(ab.age,ab.sex,color='green',linewidth=1.0,linestyle='dotted',marker='o',markerfacecolor='red')
plt.show()


# In[28]:


sns.countplot(x='sex',hue='survived',data=ab)
plt.show()


# In[29]:


ab['age'].plot(kind='hist')
plt.show()


# In[30]:


ab['fare'].value_counts()


# In[31]:


ab['pclass'].value_counts()


# In[32]:


ab['survived'].value_counts()


# In[33]:


sns.countplot(x='embarked',hue='survived',data=ab)
plt.show()


# In[34]:


sns.countplot(x='sex',hue='parch',data=ab)
plt.show()


# In[35]:


ab


# In[36]:


sns.countplot(x='sex',hue='embarked',data=ab)
plt.show()


# In[37]:


sns.countplot(x='age',hue='pclass',data=ab)
plt.show()


# In[38]:


plt.bar('sex','age',data=ab)
plt.show()


# In[39]:


ab.groupby('sex').sum()


# In[41]:


ab['sex'].value_counts()


# In[42]:


ab.shape


# In[43]:


ab.groupby('survived').sum()


# In[44]:


ab['survived'].value_counts()


# #### INSIGHTS
# 
# 
# 
# * Mostly people were travelling on fare less than 30
# * Most people were travelling on 3rd class
# * Most people were from the age of 20 to 50
# * Only 500 people survived
# * in the titanic male population was greater than female population.
# * mostly adult people survired
# * in the titanic the male is 843
# * in the ship the total female is 466
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




