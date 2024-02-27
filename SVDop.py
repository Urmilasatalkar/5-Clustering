# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 15:22:07 2023

@author: urmii
"""

import numpy as np
import pandas as pd
from numpy import array
from scipy.linalg import svd
A=array([[1,0,0,0,2],[0,0,3,0,0,],[0,0,0,0,0],[0,4,0,0,0]])
print(A)
U,d,Vt=svd(A)
print(U)
print(d)
print(Vt)
print(np.diag(d))

import pandas as pd
data=pd.read_excel('c:/2-Datasets/University_Clustering.xlsx')
data.head()
data=data.iloc[:,2:]
data
from sklearn.decomposition import TruncatedSVD
svd=TruncatedSVD(n_components=3)
svd.fit(data)
result=pd.DataFrame(svd.transform(data))
result.columns='pc0','pc1','pc2'
result.head()
