
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston


# In[10]:


boston = load_boston()
print(boston.DESCR)


# In[16]:


dataset = boston.data
for index, name in enumerate(boston.feature_names):
    print(index, name)


# In[18]:


data = dataset[:,12].reshape(-1,1)


# In[20]:


np.shape(dataset)


# In[21]:


target = boston.target.reshape(-1,1)


# In[22]:


np.shape(target)


# In[23]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data, target, color='green')
plt.xlabel('Lower income population')
plt.ylabel('Cost of house')
plt.show()


# In[24]:


from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(data, target)


# In[25]:


pred = reg.predict(data)


# In[27]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data, target, color='red')
plt.plot(data, pred, color='green')
plt.xlabel('Lower income population')
plt.ylabel('Cost of house')
plt.show()


# In[29]:


from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline


# In[38]:


model = make_pipeline(PolynomialFeatures(3), reg)


# In[39]:


model.fit(data, target)


# In[40]:


pred = model.predict(data)


# In[41]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data, target, color='red')
plt.plot(data, pred, color='green')
plt.xlabel('Lower income population')
plt.ylabel('Cost of house')
plt.show()


# In[42]:


from sklearn.metrics import r2_score


# In[43]:


r2_score(pred, target)

