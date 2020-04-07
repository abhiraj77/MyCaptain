#!/usr/bin/env python
# coding: utf-8

# In[40]:


l1=[]
l2=[]
n=int(input("Enter number of elements: "))
for i in range(0,n):
    l1.append(int(input()))
for x in l1:
    if x>0:
        l2.append(x)
print(l2)

