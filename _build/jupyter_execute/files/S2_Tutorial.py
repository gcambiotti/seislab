#!/usr/bin/env python
# coding: utf-8

# # Tutorial on [Python](https://www.python.org)
# 
# ```{contents} Sections
# :local:
# :depth: 2
# ```

# ```{div} full-width
# 
# In order to familiarize with the [Python](https://www.python.org) language you can follow an [informal introduction to Python](https://docs.python.org/3/tutorial/introduction.html) typing in the cells of your first notebook (and pressing return).
# 
# After that you can start to understand more advanced topics. 
# 
# ```

# ## Lists, tuples, sets and dictionaries
# 
# 
# ```{div} full-width
# There are four built-in data types in Python used to store collections of data: lists, tuples, sets and dictionaries, all with different qualities and usage.
# ```
# 
# ### Lists
# 
# ```{div} full-width
# 
# Lists are used to store multiple items in a single variable and are created using square brackets
# 
# ```

# In[1]:


my_list = [ 1, 1.2,"banana", [2,3] ]
print(my_list)


# ```{div} full-width
# 
# Note that the elements of a list can be of different types (integers, floats, string and lists as well).
# 
# You can unpack the elements of a list
# 
# ```

# In[2]:


a,b,c,d = my_list
print("a =",a)
print("b =",b)
print("c =",c)
print("d =",d)


# ```{div} full-width
# 
# or you can access single elements of a list using the list index inside square brackets
# 
# ```

# In[3]:


a = my_list[0]
b = my_list[2]
print("a =",a)
print("b =",b)


# ```{div} full-width
# 
# In order to know how many elements a list have you can use the function `len`
# 
# ```

# In[4]:


n = len(my_list)
print("n =",n)


# ```{div} full-width
# 
# Yoy can also create new lists which contain all the elements of a list starting from a list index or up to a list index (excluded)
# 
# ```

# In[5]:


c = my_list[2:]
d = my_list[:2]
print("c =",c)
print("d =",d)


# ```{div} full-width
# 
# In the end, you can loop over a list like this
# 
# ```

# In[6]:


for k in range(len(my_list)):
    print(my_list[k])


# ```{div} full-width
# 
# or, even better, like this
# 
# ```

# In[7]:


for elem in my_list:
    print(elem)


# ```{div} full-width
# 
# You can also have the list index and the element using the following syntax for the loop
# 
# ```

# In[8]:


for k,elem in enumerate(my_list):
    print(k,elem)


# ```{div} full-width
# 
# **Modify a list**
# 
# We can also modify a list by appending new elements 
# 
# ```

# In[9]:


my_list.append("apple")
print(my_list)


# ```{div} full-width
# 
# or modifying a specific element
# 
# ```

# In[10]:


my_list[2] = "modified element"
print(my_list)


# ```{div} full-width
# 
# Let us define a second list
# 
# ```

# In[11]:


my_second_list = [9,"banana"]
print(a)


# ```{div} full-width
# 
# and define a new list as the sum of the two already defined lists
# 
# ```

# In[12]:


my_sum_list = my_list + my_second_list 
print(my_sum_list)


# ```{div} full-width
# 
# Similar to the code in [64], you can append a new element to a list by adding a list on the fly
# 
# ```

# In[13]:


my_list += [ 47, "last element" ]
print(my_list)


# ### Tuples
# 
# ```{div} full-width
# 
# Like lists, tuples are used to store multiple items in a single variable
# 
# ```

# In[14]:


my_tuple = (1,"banana",[2,3])
print(my_tuple)


# ```{div} full-width
# 
# You can access to its elements as it was a list and loop over it
# 
# 
# ```

# In[15]:


print(my_tuple[1])


# In[16]:


for elem in my_tuple:
    print(elem)


# ```{div} full-width
# 
# The primary difference between tuples and lists is that tuples are immutable as opposed to lists which are mutable. Therefore, it is possible to change a list but not a tuple. 
# 
# When we try to modify a tuple
# 
# ```

# In[17]:


my_tuple[2] = "modified element"


# ```{div} full-width
# 
# we get an error which says that `'tuple' object does not support item assignment`. So, do not do it in your codes.
# 
# ```

# ### Dictionaries
# 
# ```{div} full-width
# Dictionaries are defined using the curly brackets and assigning a python object to a keywords as in the following examples
# ```

# In[18]:


my_dict = {"name": "john", "height": 185, "weight": 75}
print(my_dict)


# ```{div} full-width
# Afterwards, we can get the object assigned to a specific keyword by evaluating the dictionary at the desired keyword
# ```

# In[38]:


my_dict["height"]


# ```{div} full-width
# 
# We method `.keys`, `.values` and `.items` returs itarable objects listing the keywords of the dictonary (in the order with whioch they have been define), the assigned objects and tuple of keywords and assigned objects
# 
# ```

# In[39]:


print(my_dict.keys())


# In[40]:


print(my_dict.values())


# In[41]:


print(my_dict.items())


# ```{div} full-width
# 
# Here below some examples of loops over the keywords and the objects of a dictionary
# 
# ```

# In[44]:


for elem in my_dict:
    print(elem)


# In[45]:


for a in my_dict:
    print(a,my_dict[a])


# ```{div} full-width
# 
# A common way to loop over a dictionary consists in using the method `.items` as follows
# 
# ```

# In[47]:


for key,value in my_dict.items():
    print(key,value)


# ## Functions

# ```{div} full-width 
# 
# Let us understand how define functions in python from the following example
# 
# ```

# In[26]:


def myfun(x, a=0, b=0):
    y = (x-a)**2 + b
    return y


# ```{div} full-width 
# 
# Here `x` is a mandatory argument that represent the point $x$ at which we want evulate the function $(x-a)^2+b$, while `a` and `b` are optional arguments that can be used to modify the definition of the implemented function. If not given as argument when the function is called, the dafault values `a=0` and `b=0` are used. Let us use this function in some cases and print the results
# 
# ```

# In[27]:


print(myfun(4))
print(myfun(4, b=1))
print(myfun(4, a=1))
print(myfun(4, 1)) 
print(myfun(4, 1, 1)) 


# ```{div} full-width 
# 
# By the way in which it has been defined, `myfun` can accept also arrays of points $x$. Let us use the function `linspace` from `numpy` to generate an array of 5 points from -1 to 1 
# 
# ```

# In[28]:


import numpy as np
xs = np.linspace(-1,1,5)
print(xs)


# ```{div} full-width 
# 
# and evaluate the function
# 
# ```

# In[29]:


print(myfun(xs))


# ### Lambda functions

# ```{div} full-width 
# 
# We can also define a function using the `lambda` command as follows
# 
# ```

# In[30]:


a, b = 1, 1
myfun = lambda x: (x-a)**2 + b


# In[31]:


print(myfun(4))


# ```{div} full-width 
# 
# We note that the baheaviour of the lambda function changes if the two parameters are redefined
# 
# ```

# In[32]:


a, b = 0, 0


# In[33]:


print(myfun(4))


# ### Plotting
# 
# ```{div} full-width 
# In order to plot a function, let us first create a `numpy` array with the values where we intend evaluate the functions using the function `np.linspace`
# ```

# In[36]:


xs = np.linspace(-2, 2, 1001)


# ```{div} full-width
# 
# Then, we import the `matplotlib.pyplot` modules, evaluate the function and plot it
# ```

# In[37]:


import matplotlib.pyplot as plt

ys = myfun(xs)

fig, ax = plt.subplots(tight_layout=True)
ax.plot(xs,ys);


# <p style="page-break-after:always;"></p>
