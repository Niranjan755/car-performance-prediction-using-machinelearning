#!/usr/bin/env python
# coding: utf-8

# In[61]:


import numpy as np
import pandas as pd


# In[62]:


df= pd.read_csv(r"OneDrive\Desktop\Smartbridge\auto-mpg.csv")


# In[63]:


df.tail(10)


# In[64]:


df.isnull().any()


# In[65]:


x = df.iloc[:,1:8].values
y= df.iloc[:,0].values


# In[66]:


x


# In[67]:


y


# In[68]:


df["car name"].count()


# In[69]:


x.shape


# In[91]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state= 0)


# In[92]:


from sklearn.linear_model import LinearRegression
mlr = LinearRegression()
mlr.fit(x_train,y_train)


# In[93]:


ypred = mlr.predict(x_test)


# In[94]:


ypred


# In[95]:


y_test


# In[96]:


from sklearn.metrics import r2_score
accuracy = r2_score(ypred,y_test)


# In[97]:


accuracy


# In[ ]:




