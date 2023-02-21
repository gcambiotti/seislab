#!/usr/bin/env python
# coding: utf-8

# # Working with [JupyterLab](https://jupyter.org)
# 
# ```{contents} Sections
# :local:
# :depth: 2
# ```
# 
# ```{div} full-width
# 
# [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is a free minimal installer for [conda](https://docs.conda.io/en/latest). It is a small, bootstrap version of [Anaconda](https://www.anaconda.com) that includes only [conda](https://docs.conda.io/en/latest), [Python](https://www.python.org), the packages they depend on, and a small number of other useful packages, including pip, zlib and a few others. 
# 
# ```

# ## [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and [conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html)
# 
# ```{div} full-width
# 
# Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) selecting the right [Miniconda installer](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links) for your operating system and, then, create a new [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) running in a terminal
# 
# ```

# ````{div} full-width
# ```
# conda create -n seismology ipython 
# conda activate seismology  
# conda install -c conda-forge jupyter-book jupyterlab  
# conda install cartopy obspy geopy
# 
# ```
# ````

# ```{div} full-width
# 
# We have just create a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) named *seismology* and installed the packages for 
# 
# - the [IPython](https://ipython.org) interactive shell and kernel for [Jupyter](https://jupyter.org).
# - the web-based interactive development environment [JupyterLab](https://jupyter.org/) for notebooks, code, and data.
# 
# Then, we have activated the conda environment *seislab* and installed 
# 
# - [ObsPy](https://docs.obspy.org) for processing seismological data. It provides parsers for common file formats, clients to access data centers and seismological signal processing routines which allow the manipulation of seismological time series.
# - [Cartopy](https://scitools.org.uk/cartopy/docs/latest) for geospatial data processing in order to produce maps and other geospatial data analyses.
# - [GeoPy](https://geopy.readthedocs.io/en/stable/geopy) is a Python client for several popular geocoding web services that we will use just for calculating distance between geographic points.
# ```

# ```{tip}
# :class: full-width
# 
# If your operating system is Windows, please visit https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html for additional information on the installation of [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
# 
# ```

# ## Our first Jupyter notebook
# 
# ````{div} full-width
# 
# Then, we will make a directory where we will save our scripts and data, and lunch the web-based interactive development environment [JupyterLab](https://jupyter.org/) from this directory
# 
# ```
# mkdir documents/seismology
# cd documents/seismology
# jupyter lab
# ```
# 
# The last command `jupyter lab` opens a new window in your browser from where you can start a new jupyter notebook by a simple click
# 
# ![start](../images/jupyterlab_start_long.png)
# 
# type your first command in the first cell
# 
# ````

# In[1]:


print("My first command!")


# ```{div} full-width
# and save the notebook as `my_first_notebook.ipynb`
# 
# ![start](../images/jupyterlab_untitled_long.png)
# ```

# ````{admonition} Before lunching JupyterLab 
# :class: caution, full-width
# 
# Please make sure to navigate to the directory `documents/seismology`
# 
# ```
# cd documents/seismology
# ``` 
# 
# and activate the conda environment `seismology`
# 
# ```
# conda activate seismology
# ``` 
# Only then lunch the [JupyterLab](https://jupyter.org/) interface 
# 
# ```
# jupyter lab
# ```
# 
# ````

# ## Local libraries
# 
# 
# ```{div} full-width
# 
# In order to implement the python commands of this book, the local libraries `theory` and `lab` have been developped.  These are simple phyton files {Download}`theory.py<./theory.py>` and {Download}`lab.py<./lab.py>` that contain some functions that we will often use though this book, especially for plotting our results.  We have to download them and save on the directory `seismology` that we have just created and where we will work.
# 
# The content of these python files is also shown in {numref}`Lecture %s - Libraries <chap-Libraries>`.
# 
# ```

# ## Interactive computing
# 
# ```{div} full-width
# 
# Below is an example of a code cell. We will visualize some simple data using two popular packages in Python. We will use [NumPy](https://numpy.org/) to create some random data, and [Matplotlib](https://matplotlib.org) to visualize it.
# 
# ```

# In[2]:


from matplotlib import pyplot as plt
import numpy as np

# Generate 100 random data points along 3 dimensions
x, y, scale = np.random.randn(3, 100)
fig, ax = plt.subplots(tight_layout=True)

# Map each onto a scatterplot we'll create with Matplotlib
cmap = ax.scatter(x=x, y=y, c=scale, s=np.abs(scale)*500);
ax.set(title="Some random data, created with JupyterLab!");
fig.colorbar(cmap, ax=ax, label="scale");


# ```{div} full-width
# 
# You can also allow for interacting figures in order to zoom in and out freely 
# 
# ```

# In[3]:


get_ipython().run_line_magic('matplotlib', 'widget')


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ```{div} full-width
# 
# Try to run again the code cell above to see the change.
# 
# Remember also that most of the commands has an help that can be displayed. Here the help for the function `np.random.rand` that we have just used for generating 100 random data points along 3 dimensions.
# 
# ```

# In[5]:


help(np.random.rand)


# <p style="page-break-after:always;"></p>
