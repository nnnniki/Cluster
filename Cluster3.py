#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
df = pd.read_excel('~/Desktop/Online_Retail.xlsx')
df['TransactionDate']= pd.to_datetime(df['TransactionDate'])
df = df.sort_values('TransactionDate')
#删除空值 清理重复值
df1 = df.dropna()
df1 = df1.drop_duplicates()


# In[11]:


# R = 最近日期-今天日期 =对比值+1 - 购物日期
#用数据集中最大天数 再+1天 获得一个对比值 （用这个对比值取代今天日期，因为数据日期比较久）

import datetime
refrence_date = df1.TransactionDate.max() + datetime.timedelta(days = 1)
print(refrence_date)


# In[18]:


# 新增一列数据 用这个对比值-购物时间

df1['days_since_last_purchase'] = (refrence_date - df1.TransactionDate).astype('timedelta64[D]')

customer_rece = df1[['UserID', 'days_since_last_purchase']].groupby("UserID").min().reset_index()
customer_rece.rename(columns={'days_since_last_purchase':'R'}, inplace=True)
customer_rece


# In[31]:


# ID有多少条记录  F
F = df.groupby(['UserID'])['TransactionNo'].count()
Freq = pd.DataFrame(time).reset_index()
Freq.columns = ['UserID','F']
Freq


# In[19]:


# M
df1['amount'] = df1['Quan']*df1['ItemPrice']
# 每一行单价*数量
customer_mone = df1[['UserID', 'amount']].groupby("UserID").sum().reset_index()
customer_mone.rename(columns={'amount':'M'}, inplace=True)


# In[20]:


customer_mone


# In[82]:


# RFM
RF = pd.merge(customer_rece,Freq, how='inner',on='UserID')
RFM = pd.merge(RF,customer_mone, how='inner',on='UserID')
RFM


# In[84]:


RFM.drop('UserID',axis=1)


# In[106]:


# 取对数
import numpy as np
RFML = RFM.drop('UserID',axis=1)
RFML=RFML.apply(np.log,axis=1).replace(float('-inf'),0).round(2)
RFML


# In[95]:


# Z score 标准化
from sklearn import preprocessing
#建立StandardScaler对象
zscore = preprocessing.StandardScaler()
# 标准化处理
RFML_zs = zscore.fit_transform(RFML)
RFML_zs


# In[104]:


RFML_zs=RFML_zs.round(2)
RFML_zs


# In[113]:


#import pandas as pd
#data = pd.to_dataframe(RFML_zs)
#print(data.describe())

import pandas as pd
df3 = pd.DataFrame(RFML_zs)
print(df3.describe())


# In[114]:


print(df3.head(5))


# In[115]:


print(df3.isnull().any())


# In[120]:


df3=df3.fillna(0)
df3


# In[121]:


print(df3.isnull().any())


# In[134]:


# KMeans

from sklearn.cluster import KMeans

# 把数据分成3类

# 创建 Kmeans 模型并训练
k_means_model = KMeans(n_clusters = 3, random_state = 0) #设置模型参数

k_means_model.fit(df3)  #构建模型
print(k_means_model)


# In[137]:


df3['label'] = k_means_model.labels_
df3


# In[141]:


df3['label'].value_counts()


# In[139]:


df3.to_csv('~/Desktop/Cluster_result.csv',index=False)

