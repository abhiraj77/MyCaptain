# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GXaVqeXzT6r3_ZwgIInXMgij64WfXRMi
"""

!pip install numpy

import numpy as np

np.array([1,2,3])

lis = [1,2,3]
x = np.array(lis)

type(x)

lis1 = [2,4,6]
y = np.array(lis1)

x + y, x*y

z=np.zeros((7,3))

print(z)

np.ones((5,5))

np.full((3,4),8)

x=np.array([[1,2,3],[1,2,4],[2,4,6]])

z=x[2:]

print(z)

np.arange(6)

np.arange(6).reshape(2,3)

