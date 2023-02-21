#!/usr/bin/env python
# coding: utf-8

# (chap-L1)=
# # Seismic catalogs and networks
# 
# 
# ```{contents} Sections
# :local:
# :depth: 2
# ```

# ```{div} full-width
# With this lecture we will learn how download earthquake catalagos and seismic networks from the main data center of the world. As case study, we will choose the Istituto Nazionale di Geofisica e Vulcanologia (INGV) as provider and we will focus on the 2016-2017 Amatrice-Norcia-Visso seismic sequences and, in particular, on the first main shock: the 24 August 2016 M=6 Accumuli earthquake. 
# 
# In this perspective, let us import the main python libraries
# ```

# In[1]:


get_ipython().run_line_magic('matplotlib', 'widget')
import lab
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


from myst_nb import glue
get_ipython().run_line_magic('matplotlib', 'inline')
np.set_printoptions(precision=3,suppress=False)
import warnings
warnings.filterwarnings("ignore")


# In[3]:


glue(directory,"Amatrice-Norcia-Visso")


# ```{div} full-width
# and make the directory {glue:}`directory` where we will save the data that we are going to download
# ```

# In[2]:


import os

directory="Amatrice-Norcia-Visso/"
if not os.path.exists(directory):
    os.mkdir(directory)


# ## Dates

# ```{div} full-width
# In order to specify the time periods on which we are interested in and for which we want to download data, we shall use the class `UTCDateTime` from the `obspy` module
# ```

# In[4]:


from obspy import UTCDateTime


# ```{div} full-width
# Just because the Amatrice-Norcia-Visso seismic sequence spanned almost two years, from August 2016 to January 2017, we define as start time the 1-th January 2016 and end time the 1-th January 2018
# ```

# In[5]:


starttime = UTCDateTime(2016,1,1)
endtime = UTCDateTime(2018,1,1)
print("starttime:               ",starttime)
print("endtime:                 ",endtime)


# ```{div} full-width
# We note that we perform some basic operations with `UTCDateTime` objects, as for instance,
# ```

# In[58]:


newtime = starttime + 3*24*3600 + 3600+120 + 5.2
print("newtime:           ",newtime)
print("endtime-starttime: ",(endtime-starttime)/3600/24,"days")


# ## Catalogs

# ```{div} full-width
# 
# Let us import the class `Client` from `obspy` and set the Istituto Nazionale di Geofisica e Vulcanologia (INGV) as provider
# 
# ```

# In[7]:


from obspy.clients.fdsn import Client

client = Client("INGV")
print(client)


# ```{div} full-width
# Thanks to the method `client.get_events`, we will download  the earthquakes occurred in Central Italy during the 2016 and 2017 of magnitude M$\ge$5 
# ```

# In[8]:


help(client.get_events)


# ```{div} full-width
# 
# In this perspective, let us define the `file` where we will save the downloaded data and define the geographic `circle` in which we are interested in as a dictionary containing the longitude and latitude of its centre (13°,42.5°) and its radius (2°, in degrees)
# ```

# In[9]:


file = directory+"Seismic_Sequence_M5.xml"
circle = dict(maxradius=2,longitude=13,latitude=42.5)


# ```{div} full-width
# and use them as arguments of the method `client.get_events` along with the already defined `starttime` and `endtime`
# ```

# In[10]:


client.get_events(starttime=starttime, endtime=endtime, minmagnitude=5, **circle, filename=file)


# ```{div} full-width
# Afterwards, we can read the `file` and get the downloaded `Catalog` object as follows
# ```

# In[11]:


from obspy.core.event import read_events
catalog = read_events(file)
print(catalog)


# In[12]:


glue("numevents",len(catalog))


# ```{div} full-width
# As we can notice, the `catalog` contains {glue:}`numevents` earthquakes ordered by descending dates. In order to have a first understanding of what we have just downloaded, let us plot it using the the method `catalog.plot` along with the function `lab.setup_map` described in {numref}`Lecture %s Geographic maps <chap-geographic-maps>`
# ```

# In[13]:


import lab 
title = "2016-2017 Amatrice-Norcia-Visso seismic sequence (M$\geq$5)"
fig = lab.setup_map(circle=circle,color="red")
fig = catalog.plot(fig=fig,title=title)


# Here instead we zoom in the cluster of earthquakes

# In[14]:


extent = [12.5,14,42.4,43]
fig = lab.setup_map(extent=extent)
fig = catalog.plot(fig=fig,title=title)


# ### The 2016-08-24 M=6 Accumuli earthquake

# ```{div} full-width
# The `catalog` can be seen as a list of events, each of which contains information about an earthquake
# ```

# In[59]:


for ke,event in enumerate(catalog):
    print("\n\n{0:4d}-th EVENT:\n".format(ke))
    print(event)


# ```{div} full-width
# Let us select the last element of this list, which is the `Event` object describing the early earthquake of the seismic sequence
# ```

# In[16]:


event = catalog[-1]
print(event)


# ```{div} full-width
# From this `Event` object, we can get its description
# ```

# In[60]:


for event_description in event.event_descriptions:
    print(event_description.text)


# ```{div} full-width
# and its preferred origin, magnitude and focal mechanism
# ```

# In[61]:


origin = event.preferred_origin()
print("PREFERRED ORIGIN:\n\n",origin)


# In[62]:


magnitude = event.preferred_magnitude()
print("PREFERRED MAGNITUDE:\n\n",magnitude)


# In[63]:


focal_mechanism = event.preferred_focal_mechanism()
print("PREFERRED FOCAL MECHANISM:\n\n",focal_mechanism)


# ```{div} full-width
# Here a beachball plot of the focal mechanism, which show that we are considering a normal earthquake
# ```

# In[21]:


nodal_plane_1 = list(focal_mechanism.nodal_planes.get("nodal_plane_1").values())[::2]
print("nodal plane 1:",nodal_plane_1)

from obspy.imaging.beachball import beachball
fig = plt.figure(tight_layout=True)
fig = beachball(nodal_plane_1,fig=fig)


# #### Event id and additional information

# ```{div} full-width
# Let us now obtain the event id assigned to the M=6 Accumuli earthquake
# ```

# In[22]:


eventid = str(event.resource_id).split("=")[-1]
print("EVENTID:",eventid)


# ```{div} full-width
# and download a catalog with just this earthquake but with more information about it than before
# ```

# In[23]:


file = directory+eventid+"_Accumuli.xml"
client.get_events(eventid=eventid,includeallorigins=True,includeallmagnitudes=True,includearrivals=True,filename=file)
catalog = read_events(file)
print(catalog)


# In[24]:


event = catalog[0]
glue("numorigins",len(event.origins))
glue("nummagnitudes",len(event.magnitudes))


# ```{div} full-width
# Selecting the only event in the catalog, we can now see that there are {glue:}`numorigins` origins and {glue:}`nummagnitudes` magnitudes (instead of just one) and also a new field named `picks`. We will discuss this latter field in {numref}`Lecture %s <chap-L3>` {ref}`chap-L3`
# ```

# In[25]:


event = catalog[0]
print(event)


# ```{div} full-width
# In particular, we can notice that the {glue:}`numorigins` origins have different authors and values
# ```

# In[26]:


print("{0:40}  {1:>7}  {2:>7} {3:>6}  {4:>10}\n".format("author","lon","lat","dep","origin_id"))
for origin in event.origins:
    author    = origin.creation_info.author
    origin_id = origin.resource_id.id.split("=")[1]
    lon, lat  = origin.longitude, origin.latitude
    dep = origin.depth/1e3
    print("{0:40}  {1:>7.4f}  {2:>7.4f} {3:>6.2f}  {4:>10}".format(author,lon,lat,dep,origin_id))


# ```{div} full-width
# as well as the  the {glue:}`nummagnitudes` magnitudes
# ```

# In[27]:


print("{0:40}  {1:>4}  {2:>4}   {3:>10}  {4:>10} {5:>10}\n".format("Author","mag","std","type","mag_id","origin_id"))
for magnitude in event.magnitudes:
    
    author    = magnitude.creation_info.author
    
    mag_id = magnitude.resource_id.id.split("=")[1]
    origin_id = magnitude.origin_id.id.split("=")[1]
    
    mag = magnitude.mag
    mag_type = magnitude.magnitude_type
    std = magnitude.mag_errors.uncertainty
    
    if std is None: std=0
    
    print("{0:40}  {1:>4.1f}  {2:>4.1f}  {3:>5}  {4:>5}  {5:>10}".format(author,mag,std,mag_type,mag_id,origin_id))


# ```{div} full-width
# In the following, we will attempt to reproduce the results from the Bollettino Sismico Italiano by the INGV estimating the hypocenter location and origin time, as well as the local magnitude. We thus select the preferred origin that is that of Bollettino Sismico Italiano 
# ```

# In[28]:


origin = event.preferred_origin()
print("\nPREFERRED ORIGIN:\n\n",origin)


# ```{div} full-width
# and look for the associated magnitude by comparing the origin and magnitude identifiers
# ```

# In[29]:


for magnitude in event.magnitudes:
    if magnitude.origin_id == origin.resource_id: break
print("\nASSOCIATED MAGNITUDE:\n\n",magnitude)


# ```{div} full-width
# We note, indeed, that the origin identifier of the selected magnitude is the same identifier of the prefered origin. We also note that, now, the preferred origin has a new field named `arrivals`. As for the field `picks`, we will discuss this latter field in {numref}`Lecture %s <chap-L3>` {ref}`chap-L3`
# ```

# ## Inventory

# ```{div} full-width
# Let us now download all the seismic stations from all the networks managed to some extent by the INGV thank to the method `client.get_stations`. In particular, we will query the data center for all the seismic stations that have operated between the 2016 and 2017 and the information about their response functions
# ```

# In[65]:


inventory = client.get_stations(level="response",starttime=starttime,endtime=endtime)


# ```{div} full-width
# All these information are stored in an `Inventory` object.  In order to have a first understanding of what we have just downloaded, let us plot it using the the method `inventory.plot` along with the function `lab.setup_map`
# ```

# In[32]:


fig = lab.setup_map()
fig = inventory.plot(fig=fig,marker=".",size=10,color="red",label=False)


# ```{div} full-width
# Almost all stations are located in Italy, but for a few seismic station in Europe, Turkey and Marocco and one in Antartica. Here, the same plot but zoomed in Italy
# ```

# In[33]:


extent = [6.5,19,36,47.5]
fig = lab.setup_map(extent=extent)
fig = inventory.plot(fig=fig,marker="v",size=10,color="red",label=False)


# ```{div} full-width
# In order to understand what we have just downloaded, let us print the `inventory`
# ```

# In[66]:


print(inventory)


# ```{div} full-width
# Here, we print a summary of the main information of each seismic network, as the description and the starting and ending dates
# ```

# In[67]:


for network in inventory:
    
    start = str(network.start_date.date)

    if network.end_date is None: end = "present" 
    else: end = str(network.end_date.date)
    
    print("Network code: {0:2}   Period: {1:10} to {2:10}   Description: {3:60}   ".format(network.code,start,end,network.description))


# ### Seismic stations

# In[68]:


network_code, station_code = "IV", "MRLC"
glue("station",network_code+"."+station_code)


# ```{div} full-width
# Let us define the network and station codes of the seismic station {glue:}`station` that we will use as case study in this lecture
# ```

# In[69]:


network_code, station_code = "IV", "MRLC"


# ```{div} full-width
# 
# and select it from the `inventory` 
# 
# ```

# In[71]:


inv = inventory.select(network_code, station_code, time=origin.time)
print("INVENTORY:\n\n",inv)

station = inv[0][0]
print("\nSTATION:\n\n",station)


# ```{div} full-width
# As we can see, the seismic station hosts 5 different kinds of channels: HN*, HH*, BN*, LH* and VH*. The first and second letters indicates the band and instrument codes, while third letter indicates the orientation code that, in the present case, can be vertical (Z), north (N) and east (E). We can visit the web page  [SEED Channel Naming](https://ds.iris.edu/ds/nodes/dmc/data/formats/seed-channel-naming/) for a detailed description of the different band, instrument and orientation codes.
# 
# For now, let us give a look to all the channels of the seismic station
# ```

# In[40]:


for channel in station:
    print("\nCHANNEL\n\n",channel)


# ```{div} full-width
# and select some relevant information, like the sampling rate, the input and output units, and the kind of sensor associated to each channel
# ```

# In[41]:


for channel in station:
    
    code          = channel.code
    sampling_rate = channel.sample_rate
    sensor        = channel.sensor.description
    
    response = channel.response
    
    input_units  = response.instrument_sensitivity.input_units
    output_units = response.instrument_sensitivity.output_units
    
    text = r'{0:5}   sampling rate = {1:5.1f} [Hz]   input = {2:7}   output = {3:7}  {4:25}'.format(code,sampling_rate,input_units,output_units,sensor)
    print(text)


# ```{div} full-width
# 
# From this list, we can learn that all the channels convert the ground velocity (m/s) into integers (counts), but for the channels HN* which instead convert the ground acceleration (m/s$^2$) into integers (counts). We thus understand that the instrument codes H and N (the second letter) stand for velocimeters (or high gain seismometer, NANOMETRICS TRILLIUM-40S) and accelerometers (KINEMETRICS EPISENSOR-FBA-ES-T-CL-1G-FS-40-VPP), respectively. 
# 
# As it concerns the channels of the velocimeters, we can see that they are characterized by different band codes (B,H,L,V), each of which with different sampling rates. These channels comes from the same sensor (NANOMETRICS TRILLIUM-40S) and they are provided for different purposes, depending on the frequency range on which we are interested. For seismological applications, we shall select the chahnels with the highest sampling rate that, for this velocimeters, are the channels HH*.
# 
# In {numref}`Section %s - Seismometers <sec-seismometer>` we will better discuss how modern seismometers work and why they record *counts* rather than the relative motion as discussed in {numref}`Lecture %s - Harmonic oscillator <chap-H>`.
# ```

# ### Seismic channels

# ```{div} full-width
# 
# In order to dispose of an `inventory` with only the channels with the highest sampling rates, we make use of the function `lab.keep_seismic_channels` which, for each instrument codes, select the band code with the highest sampling rate
# 
# ```

# In[42]:


import lab

inventory = lab.keep_seismic_channels(inventory)
file = directory+"Seismic_Inventory.xml"
inventory.write(file,format="STATIONXML")


# ```{div} full-width
# 
# As you can see now, the seismic station {glue:}`station` only contains the HH* and HN* channels
# 
# ```

# In[43]:


inv = inventory.select(network_code, station_code, time=origin.time)
station = inv[0][0]
print("STATION:\n\n",station)


# (sec-seismometer)=
# ## Seismometers
# 
# ```{div} full-width
# A seismic station (or seismograph) consists of a sensor (seismometer),
# an analog to digital converter (digitizer) and the recorder. The
# seismometer converts the ground motion (input) into another continuous
# physical signal, like a voltage (or the motion of the stylos drawing on
# a sheet of paper). The digitizer converts the output signal from the
# sensor into a number called *count*, just like we do with a digital voltmeter.
# The digitizer is characterized by the resolution (how we discretize the
# continuous voltage into *counts*) and the sampling rate
# (how many *counts* per second). In the end, the recorder stores the
# *counts* from the ditigitizer into flash memories or hard disks.
# 
# Sensors are divided into passive and active sensors. A (modern) passive
# seismometer mainly consists of a spring-mass system and a coil embedded
# into a magnetic field that both damps the motion of the mass with
# respect to the seismometer and outputs a voltage which is nearly
# proportional to the ground velocity above the natural frequency of the
# spring-mass system (typically of the order of 1 Hz, although it
# can be as low as 0.03 Hz). It thus can be seen as a velocity
# transducer. Older passive seismometers, instead, simply consist of a
# spring-mass system in series with a dash pot (usually filled with oil)
# and measure the displacement of the mass through a stylus drawing on a
# sheet of paper or some other optical devices. An active seismometer, like the
# force balanced accelerometers (FBA), adds a displacement transducer
# (i.e., a capacitor whose capacitance varies with the displacement of the
# mass) that sends, in a negative feedback loop, a current to the coil
# which exerts a force equal and opposite to the inertia force in order to
# prevent the mass from moving at all with respect to the seismometer.
# This current, being proportional to the ground acceleration in a large
# frequency band, gives a direct measure of it.
# 
# Passive seismometers, although less accurate than the active ones, are
# cheaper and simpler to be installed into the field. 
# 
# ```
# 
# ### Velocimeters
# 
# ```{div} full-width
# A modern passive seismometer (or digital velocimeter) can be obtained using a mechanical seismometer and converting 
# the relative motion of the mass into a current through a magnetic field and a coil. 
# This current can also be used to damp the relative motion of the mass. In this case, the output 
# of the seismometer is a voltage, $V$, proportional to the relative velocity 
# of the mass with respect to the seismometer, $\dot{z}$,
# 
# $$
# \begin{align}
# &V(t) = G\,\dot{z}(t)
# \end{align}
# $$
# 
# with $G$ being the generator constant.
# The voltage is then converted into $n$ bits (typically 24), or counts, by the
# digitizer and stored into the recorder
# 
# $$
# \begin{align}
# & c(t) = C\,V(t)  
# \end{align}
# $$
# 
# The digitizer has $2^n$ ($\approx 1.7\times10^{7}$) levels that, once
# subdivided into positive and negatives levels, correspond to a dynamic
# range from $-2^{n-1}$ to $2^{n-1}-1$
# ($\approx \pm 8.4\times10^{6}$). In this respect, assuming a voltage range $[-V_\mathrm{max},V_\mathrm{max}]$, the constant $C$ reads
# 
# 
# $$
# \begin{align}
# & C = \frac{2^{n-1}}{V_\mathrm{max}}
# \end{align}
# $$
# 
# Depending on the generatot constant and on the voltage range of the voltmeter, a modern passive seismometer can be subdivides into high and low gain seismometers. High gain seismometers finely discretize a small voltage ranges, thus being accurate but prone to saturation. Low-gain seismometers, instead, are set in such a way that the voltage ranges is wide and, so, they hardly reach the saturation but they are less accurate than high-gain seismometers. The latter are use for strong ground motion applications, i.e., for studying seismic waves in proximity of the epicenter. High-gain seismometers, instead, can be used to studying teleseismic waves, far from the epicenter.
# 
# By considering the response function, as defined in  {numref}`Lecture %s <chap-H>` - {ref}`chap-H` and  {numref}`Lecture %s <chap-F>` - {ref}`chap-F`, and the fact that the output of the velocimeters is in counts, we can write the following relation between the Fourier transform of the counts and the ground velocity 
# 
# $$
# \begin{align}
# \tilde{c}(\omega) = C\,G\,\big(i\,\omega\,\tilde{z}(\omega)\big) = C\,G\,\tilde{R}(\omega)\,(i\,\omega\,\tilde{u}_g(\omega)\big)
# \end{align}
# $$
# 
# In this respect, the response function of seismometers differs by that of an harmonic oscillator only by the factor $C\,G$, that is the so called *instrument sensitivity*.
# 
# ```
# 

# ### Response function
# 
# ```{div} full-width
# Let us consider the east channel of the velocimeter (HHE) of the seismic station {glue:}`station`
# ```

# In[44]:


channel = station.select(channel="HHE")[0]


# ```{div} full-width
# and investigate its field `response`
# ```

# In[45]:


response = channel.response
print(response)


# ```{div} full-width
# As we can see, the `response` consists of $5$ stages. The first and second stages describe the actual sensor which converts the ground velocity into a voltage and the digitizer which converts the voltage into counts. The following stages, instead, are digital processings by means of which the counts are organized to be properly recorded
# ```

# In[46]:


for response_stage in response.response_stages:
    print("\nRESPONSE STAGE:\n\n",response_stage)


# ```{div} full-width
# By using the method `response.plot` we can give a first look to the Fourier transform of the response function
# ```

# In[47]:


fig,axes= plt.subplots(2, tight_layout=True, sharex=True)
fig = response.plot(0.001, output="DEF", axes=list(axes))


# ```{div} full-width
# By using the method `response.get_evalresp_response`, instead, we can evaluate the Fourier transform of the response function at the discrete frequency corresponding to a sampling time interval $[0,T)$ and step $d$
# ```

# In[48]:


T, d = 1e3, 1e-2
n = int(T/d)

Rs, fs = response.get_evalresp_response(d, n, output="DEF")

print("Frequencies:",fs)
print("Responses: ",Rs)


# ```{div} full-width
# and plot it by ourselves
# ```

# In[49]:


fig,axes = plt.subplots(2, 1, tight_layout=True, sharex=True, figsize=(5,6))
lab.plot_spectrum(fs, Rs, axes)


# ```{div} full-width
# Here below we show the response functions relating the counts to the ground displacement, velocity and acceleration
# ```

# In[53]:


fig,axes = plt.subplots(2,3,tight_layout=True,sharex=True,figsize=(12,6))
for axe,output in zip(axes.T,["DISP","VEL","ACC"]):
    Rs, fs = response.get_evalresp_response(d,n,output=output)
    ylabel = output=="DISP"
    lab.plot_spectrum(fs, Rs, axe, ylabel=ylabel)
    axe[0].set_title(output)


# ```{div} full-width
# We can also remove the last stage to the response function for understaning the impact of it 
# ```

# In[51]:


end_stage = len(response.response_stages) - 1
fig,axes = plt.subplots(2,3,tight_layout=True,sharex=True,figsize=(12,6))
for axe,output in zip(axes.T,["DISP","VEL","ACC"]):
    Rs, fs = response.get_evalresp_response(d,n,output=output,end_stage=end_stage)
    lab.plot_spectrum(fs,Rs,axe)
    axe[0].set_title(output)


# <p style="page-break-after:always;"></p>
