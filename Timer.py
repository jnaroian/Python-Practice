#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def add_time(start, duration, day='Monday'):
    startim=''
    durtim=''
    fintim=''
    new_time=''
    for char in start:
        if char.isnumeric():
            startim+=char
        elif char==':':
            startim+='.'
    for char in duration:
        if char==':':
            durtim+='.'
        else:
            durtim+=char
    fintim=float(startim)*(5/3)+float(durtim)*(5/3)
    fintim/=(5/3)
    
    fintim=(str(fintim)[:5])
    for char in str(fintim):
        if char=='.':
            new_time+=':'
        else:
            new_time+=char    

    
    return new_time
add_time("11:55 AM", "3:12")

