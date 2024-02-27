# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 23:32:35 2023

@author: urmii
"""

#3.	Analyze the information given in the following ‘Insurance Policy dataset’
# to create clusters of persons falling in the same type. 
#Refer to Insurance Dataset.csv

#business objective
'''
business objective is to perform clustering on data that have 
similar charateristics

'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


ins_df = pd.read_csv('C:/2-Datasets/Insurance Datasets.csv')
ins_df.describe()
ins_df.columns
#Index(['Premiums Paid', 'Age', 'Days to Renew', 'Claims made', 'Income'], 
#dtype='object')

ins_df.dtypes

ins_df.shape
#(100, 5)

#pairplot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(ins_df, height=3);
plt.show()

#pdf and cdf

counts, bin_edges = np.histogram(ins_df['Premiums Paid'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();

ins_df

#boxplot and outlier treatments

sns.boxplot(ins_df['Premiums Paid'])
sns.boxplot(ins_df['Age'])
sns.boxplot(ins_df['Days to Renew'])
sns.boxplot(ins_df['Claims made'])
sns.boxplot(ins_df['Income'])

# only cols premium paid, claims made have outliers

#1
iqr = ins_df['Premiums Paid'].quantile(0.75)-ins_df['Premiums Paid'].quantile(0.25)
iqr
q1=ins_df['Premiums Paid'].quantile(0.25)
q3=ins_df['Premiums Paid'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
ins_df['Premiums Paid']=  np.where(ins_df['Premiums Paid']>u_limit,u_limit,np.where(ins_df['Premiums Paid']<l_limit,l_limit,ins_df['Premiums Paid']))
sns.boxplot(ins_df['Premiums Paid'])

#2
iqr = ins_df['Claims made'].quantile(0.75)-ins_df['Claims made'].quantile(0.25)
iqr
q1=ins_df['Claims made'].quantile(0.25)
q3=ins_df['Claims made'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
ins_df['Claims made']=  np.where(ins_df['Claims made']>u_limit,u_limit,np.where(ins_df['Claims made']<l_limit,l_limit,ins_df['Claims made']))
sns.boxplot(ins_df['Claims made'])


ins_df.describe()
#we can see that there is huge difference between min,max and mean
# values for all the columns so we need to normalize the dataset

def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return x

df_normal = norm_func(ins_df)
desc = df_normal.describe()
desc
df_normal.columns

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z = linkage(df_normal,method='complete',metric='euclidean')
plt.figure(figsize=(15,8))
plt.title('Hierarchical clustering dendrogram')
plt.xlabel('index')
plt.ylabel('Distance')
#ref of dendrogram

sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()


#now apply clustering 
from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters=3,linkage='complete',affinity='euclidean').fit(df_normal)
#apply labels to clusters
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
#assign this series to autoIns dataframe as column
ins_df['cluster'] = cluster_labels
ins_df.columns
ins_df.shape
ins_dfNew = ins_df.iloc[:,[-1,0,1,2,3,4,5,6,7,8,9,10]]
ins_dfNew.columns

ins_dfNew.iloc[:,2:].groupby(ins_dfNew.cluster).mean()
ins_dfNew.to_csv("Insurance DatasetNew.csv",encoding='utf-8')
ins_dfNew.cluster.value_counts()
import os
os.getcwd()

#*******************************************************

#kmeans clustering on insurance data
#for this we will use normalized dataset i.e df_normal

from sklearn.cluster import KMeans
#total sum of squares
TWSS = []

#initially we will find the ideal cluster number using elbow curve

k = list(range(2,8))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_normal)
    TWSS.append(kmeans.inertia_)
  
TWSS
'''
[21.37095544074799,
 15.758410404424241,
 12.06613037075154,
 9.729441304368878,
 8.101284801872284,
 7.230049855975996]
'''

'''
k selected by calculating the difference or decrease in
twss value 
'''
def find_cluster_number(TWSS):
    diff =[]
    for i in range(0,len(TWSS)-1):
        d = TWSS[i]-TWSS[i+1]
        diff.append(d)
    max = 0
    k =0
    for i in range(0,len(diff)):
        if max<diff[i]:
            max = diff[i]
            k = i+3
    return k

k = find_cluster_number(TWSS)
print("Cluster number is = ",k)
plt.plot(k,TWSS,'ro-')
plt.xlabel('No of clusters')
plt.ylabel('Total_within_SS')

model = KMeans(n_clusters=k)
model.fit(df_normal)
model.labels_
mb = pd.Series(model.labels_)
df_normal['clusters'] = mb
df_normal.head()
df_normal.shape
df_normal.columns
df_normal = df_normal.iloc[:,[-1,0,1,2,3,4]]
df_normal
#df_normal.drop(['clusters'],axis=1,inplace=True)
df_normal.iloc[:,2:5].groupby(df_normal.clusters).mean()
df_normal.to_csv('C:/4-data mining/k_means_Insurance_data.csv')
import os
os.getcwd()



