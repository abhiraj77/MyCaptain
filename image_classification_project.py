
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


data = pd.read_csv('mnist_test.csv')


# In[5]:


data.head()


# In[6]:


a = data.iloc[4,1:].values


# In[7]:


a = a.reshape(28,28).astype('uint8')
plt.imshow(a)


# In[9]:


df_x = data.iloc[:,1:]
df_y = data.iloc[:,0]


# In[10]:


x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2, random_state=4)


# In[13]:


x_train.head()


# In[14]:


y_train.head()


# In[16]:


rf = RandomForestClassifier(n_estimators=100)


# In[17]:


rf.fit(x_train,y_train)


# In[18]:


pred = rf.predict(x_test)


# In[19]:


pred


# In[21]:


s = y_test.values

count = 0
for i in range(len(pred)):
    if pred[i] == s[i]:
        count+=1


# In[22]:


count


# In[23]:


len(pred)


# In[24]:


count/len(pred)

