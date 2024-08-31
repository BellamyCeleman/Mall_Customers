#!/usr/bin/env python
# coding: utf-8

# In[67]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[68]:


df = pd.read_csv('Mall_Customers.csv')


# In[69]:


df.head()


# In[70]:


df = df.drop('CustomerID', axis=1)


# In[71]:


df.describe()


# In[72]:


df.info()


# In[73]:


df['Genre'].value_counts()


# In[74]:


sns.countplot(df, x='Genre', hue='Genre', palette='dark')


# In[75]:


df['Genre'] = df['Genre'].replace({'Male': 0, 'Female': 1}).astype(int)


# In[76]:


df.head()


# In[77]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler

sc = StandardScaler()

X = sc.fit_transform(df)


# # Elbow method

# In[78]:


from sklearn.cluster import KMeans

ssd = []

for k in range(2, 10):
    model = KMeans(n_clusters=k)
    model.fit(X)

    ssd.append(model.inertia_)

plt.plot(range(2, 10), ssd)


# In[ ]:





# In[79]:


model = KMeans(n_clusters=5)
lables = model.fit_predict(X)


# In[81]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=lables, palette='viridis', s=65, edgecolor='black', alpha=0.85)


# In[ ]:




