#!/usr/bin/env python
# coding: utf-8

# In[37]:


a=c=0;b=1
print(str(a)+"\n"+str(b))
while c<=1000:
    c=a+b
    a=b
    b=c
    print(c)

