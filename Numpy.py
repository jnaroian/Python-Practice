#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Creating Arrays
import numpy as np


# In[3]:


my_list1 =[1,2,3,4]


# In[4]:


my_array1=np.array(my_list1)
#making an array takes a list as input


# In[5]:


my_list2=[11,22,33,44]


# In[9]:


my_lists=[my_list1,my_list2]


# In[10]:


my_array2=np.array(my_lists)


# In[11]:


my_array2
#multidimensional array


# In[12]:


my_array2.shape


# In[13]:


my_array2.dtype
#shows data type


# In[15]:


np.zeros(5)


# In[16]:


my_zeros_array=np.zeros(5)
my_zeros_array.dtype


# In[17]:


np.ones([5,5])


# In[19]:


np.empty(5)
#same as np.zeros


# In[20]:


np.eye(5)
#identity matrix, always square


# In[21]:


np.arange(5,50,2)
#(start, stop, step)


# In[22]:


#Using Arrays and Scalars


# In[27]:


arr1=np.array([[1,2,3,4],[8,9,10,11]])


# In[29]:


arr1


# In[30]:


arr1*arr1


# In[31]:


arr1-arr1


# In[32]:


1/arr1


# In[33]:


arr1 ** 3


# In[34]:


#Indexing arrays


# In[36]:


arr=np.arange(0,11)


# In[37]:


arr


# In[38]:


arr[8]
#calls index at 8


# In[41]:


arr[1:5]


# In[42]:


arr[0:5]


# In[44]:


arr[0:5]=100
arr


# In[46]:


arr=np.arange(0,11)
arr


# In[48]:


slice_of_arr = arr[0:6]
slice_of_arr


# In[49]:


slice_of_arr[:]=99
slice_of_arr


# In[50]:


arr
#changes to slice, changes original


# In[58]:


#if you don't want that, make copy
slice_of_arr=arr[0:6].copy()
slice_of_arr[:]=1
slice_of_arr


# In[59]:


arr
#see, stays the same now


# In[60]:


arr_2d = np.array(([5,10,15],[20,25,30],[35,40,45]))
arr_2d


# In[61]:


arr_2d[1][2]
#array[row][column]


# In[64]:


#slice 2d array
arr_2d_slice=arr_2d[:2,1:]
arr_2d_slice


# In[74]:


arr2d=np.zeros([10,10])
arr2d


# In[75]:


arr_length=arr2d.shape[1]


# In[76]:


arr_length


# In[77]:


for i in range(arr_length):
   arr2d[i]=i
arr2d


# In[78]:


#fancy indexing
arr2d[[2,4,6,8]]


# In[79]:


arr2d[[6,4,2,7]]


# In[80]:


#Array Transposition


# In[82]:


arr=np.arange(50).reshape([10,5])
arr


# In[83]:


arr.T
#transposes matrix


# In[84]:


np.dot(arr.T,arr)
#dot products transposed array with original


# In[86]:


arr3d= np.arange(50).reshape((5,5,2))
arr3d
#3d matrix


# In[87]:


arr3d.transpose((1,0,2))
#This transpose makes the corresponding rows and columns of each slice, together in thier own slice


# In[94]:


arr= np.array([[1,2,3]])
arr


# In[95]:


arr.swapaxes(0,1)
#swapaxes(swaps this axes, for this axes)


# In[96]:


#Universal Array Functions


# In[97]:


arr=np.arange(0,11)
arr


# In[98]:


np.sqrt(arr)
#takes square root of every value


# In[99]:


np.exp(arr)
#takes e to the power of every value in array


# In[100]:


A=np.random.randn(10)
A
#np.random calls randim function, randn calls normal distribution of random numbers


# In[101]:


B=np.random.randn(10)
B


# In[102]:


#binary functions
np.add(A,B)


# In[104]:


np.maximum(A,B)
#finds maximum between two arrays, and picks the higher value for the array of the same shape


# In[106]:


#for more functions look up website
website = 'http://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs'
import webbrowser
#webbrowser.open(website)
#run that ^


# In[107]:


#Array processing


# In[108]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#call to see plots in notebook


# In[110]:


points = np.arange(-5,5,0.01)
#array of points


# In[111]:


dx,dy = np.meshgrid(points,points)
#meshgrid returns coordinate points


# In[112]:


dx


# In[113]:


dy


# In[114]:


z=(np.sin(dx)+np.sin(dy))
z


# In[115]:


#plotting points, imshow
plt.imshow(z)


# In[116]:


#add color bar and title
plt.imshow(z)
plt.colorbar()
plt.title('Plot for sin(x)+sin(y)')


# In[117]:


#List Comprehension
A = np.array([1,2,3,4])
B = np.array([100,200,300,400])


# In[118]:


#make boolean array, for list comprehension
condition = np.array([True,True, False,False])


# In[119]:


#Make something to chose A value when condition is true, otherwise choose B value
answer = [(A_val if cond else B_val) for A_val, B_val, cond in zip(A,B,condition)]


# In[120]:


answer


# In[122]:


#numpy.where
answer2= np.where(condition,A,B)
#short for of above thing
answer2


# In[123]:


from numpy.random import randn


# In[125]:


arr = randn(5,5)
arr


# In[126]:


np.where(arr<0,0,arr)
#where the array is less than 0 (condition), replace with zero, otherwise replace with arr value


# In[127]:


arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
arr


# In[128]:


arr.sum()


# In[129]:


arr.sum(0)
#sums along zero axis, adds up columns


# In[130]:


arr.mean()


# In[131]:


arr.std()
#standard dev


# In[132]:


arr.var()
#variance


# In[133]:


bool_arr=np.array([True,False,True])
bool_arr.any()
#returns true if anything in the array is true


# In[134]:


bool_arr.all()
#returns true only if all array is true


# In[135]:



arr = randn(5)
arr


# In[136]:


#sorting array
arr.sort()
arr


# In[137]:


countries=np.array(['France','Germany','USA','Russia','USA','Mexico','Germany'])


# In[138]:


np.unique(countries)
#tells you each value once only


# In[139]:


np.in1d(['France','USA','Sweden'],countries)
#checks if in1d(objects in this array, are in this array)


# In[140]:


#Array Input and Output


# In[141]:


arr=np.arange(5)
arr


# In[143]:


np.save('myarray',arr)
#saves an array with that name


# In[144]:


arr=np.arange(10)
arr


# In[145]:


np.load('myarray.npy')
#loads saved array, add .npy file to end to load it


# In[146]:


arr1=np.load('myarray.npy')
arr1


# In[148]:


arr2=arr
arr2


# In[152]:


np.savez('ziparray',x=arr1,y=arr2)
#saves arrays as zip file 


# In[153]:


archive_array=np.load('ziparray.npz')
archive_array['x']


# In[205]:


arr=np.array([[1,2,3],[4,5,6]])
arr


# In[206]:


np.savetxt('mytextarray',arr,delimiter=',')
#saves as text file instead of array, delimiter separates the numbers


# In[207]:


arrtxt=np.loadtxt('mytextarray',delimiter=',')
arrtxt


# In[208]:


#Dataframe


# In[ ]:




