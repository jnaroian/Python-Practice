#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Series
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
#Means you don't have to write pd.series or pd.dataframe all the time


# In[3]:


#Series indexes values, they are numbered
obj = Series([3,6,9,12])
obj


# In[4]:


#If you wanna see values as array
obj.values


# In[5]:


obj.index
#shows series index


# In[6]:


#name the index
ww2_cas=Series([87000,43000,30000,21000,4000], index=['USSR','Germany','China','Japan','USA'])
ww2_cas


# In[7]:


ww2_cas['USA']


# In[8]:


#To check which countries has casualties over a certain number for example
ww2_cas[ww2_cas>40000]


# In[9]:


'USSR' in ww2_cas


# In[10]:


#Convert series into dictionary
ww2_dict=ww2_cas.to_dict()
ww2_dict


# In[11]:


ww2_series=Series(ww2_dict)
ww2_series


# In[12]:


countries=['China','Germany','Japan','USA','USSR','Argentina']


# In[14]:


obj2=Series(ww2_dict,index=countries)
#setting index to another list in series
obj2


# In[15]:


pd.isnull(obj2)
#tells you if there is a null value


# In[16]:


pd.notnull(obj2)


# In[17]:


ww2_series + obj2


# In[18]:


#name values
obj2.name="WW2 Casualties"
obj2


# In[20]:


#name index
obj2.index.name='Countries'
obj2


# In[21]:


#Dataframe


# In[32]:


#data copied from nfl win loss record wikipedia, copy and write this
nfl_frame= pd.read_clipboard()
nfl_frame


# In[31]:


nfl_frame.columns
#gives you column names


# In[29]:


nfl_frame['Dallas Cowboys']


# In[33]:


DataFrame(nfl_frame,columns=['Dallas Cowboys','947','538'])
#make new dataframe with selected columns


# In[34]:


DataFrame(nfl_frame, columns=['Dallas Cowboys','947','538','Stadium'])
#add new column that doesn't exist yet


# In[39]:


#retrieve rows instead, head(number of first rows you want), tail(number of last rows you want)
nfl_frame.head(3)


# In[46]:


nfl_frame.loc[3]
#retrieve specific index row


# In[47]:


nfl_frame['Stadium']='Levis Stadium'
nfl_frame


# In[51]:


#makes column an array
nfl_frame["Stadium"]=np.arange(4)
nfl_frame


# In[54]:


#make series
stadiums=Series(['Levis Stadium','AT&T Stadium'], index=[3,0])
stadiums


# In[55]:


nfl_frame['Stadium']=stadiums
#set df column to the series
nfl_frame


# In[56]:


#delete entire columns
del nfl_frame['Stadium']
nfl_frame


# In[57]:


data ={'City':['SF','LA','NYC'],'Polulation':[837000,3880000,840000]}


# In[58]:


city_frame=DataFrame(data)
city_frame


# In[59]:


#Index Objects


# In[60]:


my_ser=Series([1,2,3,4],index=['A','B','C','D'])
my_ser


# In[61]:


my_index=my_ser.index
my_index


# In[63]:


my_index[2:]


# In[64]:


my_index[0]='W'
#cannot change indexes like this


# In[65]:


#Reindexing


# In[66]:


from numpy.random import randn


# In[67]:


ser1=Series([1,2,3,4],index=['A','B','C','D'])
ser1


# In[69]:


ser2=ser1.reindex(['A','B','C','D','E','F'])
ser2
#reindexing


# In[71]:


ser2.reindex(['A','B','C','D','E','F','G'],fill_value=0)
#makes new indexes filled with input value


# In[73]:


ser3=Series(['USA','Mexico','Canada'],index=[0,5,10])
ser3


# In[75]:


ranger=range(15)
#list up to 15
ser3.reindex(ranger,method='ffill')
#ffill means forward fill, reindex(new index, method = method of fill)


# In[78]:


dframe=DataFrame(randn(25).reshape((5,5)),index=['A','B','D','E','F'],columns=['col1','col2','col3','col4','col5'])
dframe
#making dataframe from scratch


# In[80]:


dframe2=dframe.reindex(['A','B','C','D','E','F'])
dframe2
#add index row


# In[81]:


new_columns = ['col1','col2','col3','col4','col5','col6']
dframe2.reindex(columns=new_columns)
#reindex(columns=the list of updated columns)


# In[88]:


#Drop Entry


# In[90]:


ser1=Series(np.arange(3),index=['a','b','c'])
ser1


# In[91]:


#drop an index from series
ser1.drop('b')


# In[92]:


dframe1=DataFrame(np.arange(9).reshape((3,3)),index=['SF','LA','NY'],columns=['pop','size','year'])
dframe1


# In[95]:


dframe2=dframe1.drop('LA')
dframe2


# In[103]:


dframe1.drop('year',axis=1)
#When dropping columns, you need to specify axis=1, axis=0 is default and those are rows


# In[104]:


#Selecting Entries


# In[105]:


ser1=Series(np.arange(3), index=['A','B','C'])
ser1=2*ser1
#Doubled every value in series
ser1


# In[106]:


ser1['B']
#Call 2 with index


# In[107]:


ser1[1]
#Calls number of index, so this would be 2


# In[108]:


ser1[['A','B']]


# In[109]:


ser1[ser1>3]


# In[111]:


dframe=DataFrame(np.arange(25).reshape((5,5)), index=['NY','LA','SF','DC','Chi'], columns= ['A','B','C','D','E'])
dframe


# In[114]:


dframe[['B','E']]


# In[115]:


dframe[dframe['C']>8]


# In[116]:


dframe>10


# In[117]:


dframe.loc['LA']


# In[119]:


dframe.iloc[1]


# In[120]:


#Data alignment


# In[125]:


ser1=Series([0,1,2], index=['A','B','C'])
ser1


# In[126]:


ser2=Series([3,4,5,6], index = ['A','B','C','D'])
ser2


# In[127]:


ser1+ser2
#Add 2 unequal series together


# In[129]:


dframe1=DataFrame(np.arange(4).reshape((2,2)),columns=list('AB'),index=['NYC','LA'])
dframe1


# In[133]:


dframe2=DataFrame(np.arange(9).reshape((3,3)),columns=list('ADC'),index=['NYC','SF','LA'])
dframe2


# In[135]:


dframe1+dframe2
#Same thing as before, only where row and column match up does it work


# In[136]:


dframe1.add(dframe2,fill_value=0)
#Fills in values of dframe1 as zero, adds dframe values to it


# In[137]:


ser3=dframe2.iloc[0]
ser3


# In[138]:


dframe2-ser3


# In[139]:


#Ranking and Sorting


# In[140]:


ser1=Series(range(3),index=['C','A','B'])
ser1


# In[142]:


ser1.sort_index()
#sorts index values


# In[144]:


ser1.sort_values()
#sorts by values


# In[145]:


ser2=Series(randn(10))
ser2


# In[146]:


ser2.sort_values()


# In[147]:


#Each value is ranked, to know the rank do this
ser2.rank()


# In[149]:


ser3=Series(randn(10))
ser3


# In[150]:


ser3.rank()


# In[152]:


ser3.sort_values()


# In[153]:


ser3.rank()


# In[154]:


#Summary Statistics


# In[156]:


arr=np.array([[1,2,np.nan],[np.nan,3,4]])
arr


# In[157]:


dframe1=DataFrame(arr, index=['A','B'],columns=['One','Two','Three'])
dframe1


# In[158]:


dframe1.sum()


# In[159]:


dframe1.sum(axis=1)
#sums rows


# In[161]:


dframe1.min(axis=1)


# In[164]:


dframe1.idxmin(axis=0)
#gives index of min value


# In[169]:


dframe1.cumsum()
#cumulation sum of columns, for rows use axis=1


# In[170]:


dframe1.describe()


# In[179]:


from pandas_datareader import data as pdweb
import datetime
#gets info off web and imports the date and time


# In[180]:


prices = pdweb.get_data_yahoo(['CVX','XNM','BP'], start=datetime.datetime(2010,1,1),end=datetime.datetime(2013,1,1))['Adj Close']
prices.head()
#Lets you get data off the internet


# In[181]:


volume = pdweb.get_data_yahoo (['CVX','XNM','BP'], start=datetime.datetime(2010,1,1),end=datetime.datetime(2013,1,1))['Volume']
volume.head()
#Makes dataframe of volume


# In[182]:


rets= prices.pct_change()
#returns on stocks


# In[183]:


#correlation of the stocks
corr=rets.corr


# In[184]:


get_ipython().run_line_magic('matplotlib', 'inline')
prices.plot()
#plots prices, first line makes plot visible


# In[185]:


import seaborn as sns
#plotting library
import matplotlib.pyplot as plt


# In[186]:


sns.corrplot(rets,annot=false,diag_names=false)


# In[187]:


ser1=Series(['w','w','x','y','z','w','x','y','x','a'])
ser1


# In[188]:


ser1.unique()
#returns array with unique values


# In[189]:


ser1.value_counts()


# In[190]:


#Missing Data


# In[191]:


data=Series(['one','two',np.nan,'four'])
data


# In[192]:


data.isnull()
#tells you if a value is null


# In[193]:


data.dropna()
#removes any null value


# In[194]:


dframe=DataFrame([[1,2,3],[np.nan,5,6],[7,np.nan,9], [np.nan,np.nan,np.nan]])
dframe


# In[195]:


clean_frame = dframe.dropna()
#works the same way in dataframe
clean_frame


# In[196]:


dframe.dropna(how='all')
#drops rows where all data is null


# In[198]:


dframe.dropna(how='all',axis=1)
#drops column with all null values


# In[199]:


npn=np.nan
dframe2=DataFrame([[1,2,3,npn],[2,npn,5,6],[npn,7,npn,9],[1,npn,npn,npn]])
dframe2


# In[200]:


dframe2.dropna(thresh=2)
#removes rows with under 2 datapoints, 2 points is the threshhold


# In[201]:


dframe2.fillna(1)
#fills null values with what's in the parenthesis


# In[202]:


dframe2


# In[203]:


dframe2.fillna({0:0,1:1,2:2,3:3})
#fill different columns with different values by using dictionary


# In[204]:


#Index Hierarchy


# In[206]:


ser=Series(randn(6),index=[[1,1,1,2,2,2],['a','b','c','a','b','c']])
ser
#when 2 indexes are used, pandas gives you levels of indexes, with subgroups made


# In[207]:


ser[1]


# In[210]:


#to index a subgroup, 'a' for example, use colon 
ser[:,'a']


# In[212]:


dframe= ser.unstack()
dframe
#creates data frame out of multi indexed series


# In[220]:


dframe2=DataFrame(np.arange(16).reshape(4,4),index=[['a','a','b','b'],[1,2,1,2]], columns=[['NY','NY','SF','LA'],['cold','hot','hot','cold']])
dframe2
#double indexed on both axis


# In[221]:


dframe2.index.names=['INDEX_1','INDEX_2']
dframe2.columns.names=['Cities','Temp']
dframe2
#names indexes


# In[224]:


dframe2.swaplevel('Cities','Temp',axis=1)
#changes the hierarchy, whats on top and bottom


# In[229]:


dframe2


# In[228]:


dframe2.groupby(level='Temp',axis=1).sum()
#sums up the rows in a chosen hierarchy, in this case, Temp


# In[ ]:




