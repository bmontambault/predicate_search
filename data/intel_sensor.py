
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ### Process original data

# original sensor data data: http://db.csail.mit.edu/labdata/labdata.html
# 
# download original sensor data: https://tufts.box.com/s/n3k4iunl8a9wzhsnbxnconss8qx8kxfy

# In[3]:


intel_sensor_original = pd.read_csv('intel_sensor_original.txt', delimiter=" ", header=None)


# In[4]:


intel_sensor_original.columns = ['date', 'time', 'epoch', 'moteid', 'temperature', 'humidity', 'light', 'voltage']


# In[5]:


intel_sensor_original = intel_sensor_original[~np.any(intel_sensor_original.isnull(), axis=1)]


# In[6]:


intel_sensor_original['datetime'] = pd.to_datetime(intel_sensor_original.date.astype(str) + ' ' + intel_sensor_original.time.astype(str))


# In[7]:


intel_sensor_original = intel_sensor_original.drop(['date', 'time'], axis=1)


# In[8]:


intel_sensor_original.head()


# In[8]:


intel_sensor_original.to_csv('intel_sensor.csv', index=False)


# ### Scenario 1

# sensor 15 starts dying and generating temperatures above 100

# download sensor data: https://tufts.box.com/s/br0938bp02u4y9pthju9aj0gol8hl8ye

# In[9]:


intel_sensor = pd.read_csv('intel_sensor.csv')
intel_sensor.datetime = pd.to_datetime(intel_sensor.datetime)


# In[10]:


s1_startdate = '2004-2-29'
s1_enddate = '2004-3-3'


# In[11]:


intel_sensor_s1 = intel_sensor[(intel_sensor.datetime>=s1_startdate) & (intel_sensor.datetime<=s1_enddate)]


# All sensor measurements

# In[12]:


intel_sensor_s1.groupby('datetime').temperature.max().plot()


# Measurements for sensor 15

# In[18]:


intel_sensor_s1[intel_sensor_s1.moteid == 15].groupby('datetime').temperature.max().plot()


# In[19]:


intel_sensor_s1.to_csv('intel_sensor_s1.csv', index=False)


# Download scenario 1 data: https://tufts.box.com/s/iqq1m7vid64sqp1qdxo7pxrgk5r76epq

# ### Scenario 2

# sensor 18 starts to lose battery power (indicated by low voltage) and generating temperatures above 100
# 
# download sensor data: https://tufts.box.com/s/br0938bp02u4y9pthju9aj0gol8hl8ye

# In[21]:


intel_sensor = pd.read_csv('intel_sensor.csv')
intel_sensor.datetime = pd.to_datetime(intel_sensor.datetime)


# In[34]:


s2_startdate = '2004-3-7'
s2_enddate = '2004-3-15'


# In[35]:


intel_sensor_s2 = intel_sensor[(intel_sensor.datetime>=s2_startdate) & (intel_sensor.datetime<=s2_enddate)]


# All sensor measurements

# In[36]:


intel_sensor_s2.groupby('datetime').temperature.max().plot()


# Measurements for sensor 18

# In[38]:


intel_sensor_s2[intel_sensor_s2.moteid == 18].groupby('datetime').temperature.max().plot()


# In[39]:


intel_sensor_s2.to_csv('intel_sensor_s2.csv', index=False)


# Download scenario 2 data: https://tufts.box.com/s/kak5r8lnj37hz5y0s0d2hrppfoljojt2

# ### Scenario 3
# 
# Extension of scenario 1. Sensor 15 starts dying and generating temperatures above 100. Sensor 20 generates low readings on the 28th.
# 
# download sensor data: 

# In[3]:


intel_sensor = pd.read_csv('intel_sensor.csv')
intel_sensor.datetime = pd.to_datetime(intel_sensor.datetime)


# In[17]:


s3_startdate = '2004-2-28'
s3_enddate = '2004-3-3'


# In[18]:


intel_sensor_s3 = intel_sensor[(intel_sensor.datetime>=s3_startdate) & (intel_sensor.datetime<=s3_enddate)]


# In[19]:


intel_sensor_s3.groupby('datetime').temperature.max().plot()


# In[21]:


intel_sensor_s3.to_csv('intel_sensor_s3.csv', index=False)


# Download scenario 3 data: https://tufts.box.com/s/x9q0xdc5f9hl7jzquwdj7mm6m9lii2pi
