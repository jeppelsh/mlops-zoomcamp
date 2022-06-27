#!/usr/bin/env python
# coding: utf-8

# In[15]:


get_ipython().system('pip freeze | grep scikit-learn')


# In[4]:


get_ipython().system('pip install scikit-learn==1.0.2')


# In[30]:


import pickle
import pandas as pd
import sklearn
import pyarrow


# In[21]:


with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


# In[22]:


categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[23]:


df = read_data('https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_2021-02.parquet')


# In[24]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)


# In[11]:


y_pred.mean()


# In[25]:


year = 2021
month = 2


# In[26]:


df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# In[28]:


df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['predictions'] = y_pred


# In[31]:


output_file = "result.parquet"


# In[32]:


df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)


# In[35]:


get_ipython().system('ls -hl |grep result.parquet')


# In[ ]:




