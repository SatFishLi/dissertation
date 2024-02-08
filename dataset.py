
# coding: utf-8

# In[1]:
import pandas as pd
import numpy as np


# In[2]:

# guagua is the name of the file
data = pd.read_excel('data/guagua(3).xlsx',index_col=0)
data.head()


# In[3]:


data.columns


# In[4]:


unique_Geomorphology = data['Geomorphology'].unique()
print(unique_Geomorphology)


# In[5]:


unique_Natural = data['Natural habitat'].unique()
print(unique_Natural)


# In[6]:


#Geomorphology
data['Geomorphology'] = data['Geomorphology'].replace(['rocky', 'high cliff', 'Seawalls'], 1)
data['Geomorphology'] = data['Geomorphology'].replace(['Medium Cliff', 'indented coast', 'small seawalls'], 2)
data['Geomorphology'] = data['Geomorphology'].replace(['Low cliff', 'beachrocks'], 3)
data['Geomorphology'] = data['Geomorphology'].replace(['Cobble beach', 'estuary', 'lagoon','bluff'], 4)
data['Geomorphology'] = data['Geomorphology'].replace(['sand', 'beach'], 5)


# In[7]:


#Elevation
data['Z(DTM1m_2022)'] = np.where(data['Z(DTM1m_2022)'] > 35, 1,
                                np.where(data['Z(DTM1m_2022)'] >= 9, 2,
                                         np.where(data['Z(DTM1m_2022)'] >= 6, 3,
                                                  np.where(data['Z(DTM1m_2022)'] >= 3, 4, 5))))


# In[8]:


#Natural Habitats
data['Natural habitat'] = data['Natural habitat'].replace(['Coral reef', 'Mangrove', 'forest'], 1)
data['Natural habitat'] = data['Natural habitat'].replace(['high dune', 'Coastal saltmarsh','marsh'], 2)
data['Natural habitat'] = data['Natural habitat'].replace(['low dune'], 3)
data['Natural habitat'] = data['Natural habitat'].replace(['seagrass','Kelp'], 4)
data['Natural habitat'] = data['Natural habitat'].replace(['no habitat'], 5)


# In[9]:


#SLR
data['SLR'] = np.where(data['SLR'] < 0.5, 1,
                      np.where((data['SLR'] >= 0.5) & (data['SLR'] < 1.5), 2,
                               np.where((data['SLR'] >= 1.5) & (data['SLR'] < 2.5), 3,
                                        np.where((data['SLR'] >= 2.5) & (data['SLR'] < 3.5), 4, 5))))


# In[10]:


#WAVE
data['WAVE'] = np.where(data['WAVE'] < 1, 1,
                      np.where((data['WAVE'] >= 1) & (data['WAVE'] < 3), 2,
                               np.where((data['WAVE'] >= 3) & (data['WAVE'] < 4), 3,
                                        np.where((data['WAVE'] >= 4) & (data['WAVE'] < 5), 4, 5))))


# In[11]:


#SLOPE%
data['SLOPE%'] = np.where(data['SLOPE%'] > 20, 1,
                      np.where((data['SLOPE%'] >= 12) & (data['SLOPE%'] < 20), 2,
                               np.where((data['SLOPE%'] >= 7) & (data['SLOPE%'] < 12), 3,
                                        np.where((data['SLOPE%'] >= 3) & (data['SLOPE%'] < 7), 4, 5))))


# In[12]:


#TIDE
data['TIDE'] = np.where(data['TIDE'] > 5, 1,
                      np.where((data['TIDE'] >= 3.5) & (data['TIDE'] < 5), 2,
                               np.where((data['TIDE'] >= 2) & (data['TIDE'] < 3.5), 3,
                                        np.where((data['TIDE'] >= 1) & (data['TIDE'] < 2), 4, 5))))


# In[13]:


#EPR1
data['EPR1'] = np.where(data['EPR1'] > 2, 1,
                      np.where((data['EPR1'] >= 1) & (data['EPR1'] < 2), 2,
                               np.where((data['EPR1'] >= -1) & (data['EPR1'] < 1), 3,
                                        np.where((data['EPR1'] >= -2) & (data['EPR1'] < -1), 4, 5))))


# In[14]:


data = data.drop(columns=['ten_riley','SLOPE','EPR2'])


# In[15]:


# Show the size of the values in each column
data['Natural habitat'].values


# In[16]:


data.head()


# In[17]:


# Calculate the value of the CVI column
data['CVI'] = (data.iloc[:, :8].prod(axis=1)/ 8)**0.5
data.head()


# In[18]:


# Getting the maximum and minimum values of the CVI
cvi_min = data['CVI'].min()
cvi_max = data['CVI'].max()

# Calculate the range of the interval
interval = (cvi_max - cvi_min) / 5

# Creating a list of tags
labels = ['class1', 'class2', 'class3', 'class4', 'class5']

data['TARGET '] = pd.cut(data['CVI'], bins=[cvi_min-1, cvi_min+interval, cvi_min+2*interval,
                                           cvi_min+3*interval, cvi_min+4*interval, cvi_max],
                        labels=labels)

# View Results
data


# In[19]:


data.to_excel('data/data.xlsx')

