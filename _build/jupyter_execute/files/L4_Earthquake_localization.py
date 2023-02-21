#!/usr/bin/env python
# coding: utf-8

# (chap-L3)=
# # Earthquake localization
# 
# ```{contents} Sections
# :local:
# :depth: 2
# ```
# 
# ```{div} full-width
# 
# In this lesson, we will locate the origin of the 24/8/2016 M=6 Accumoli earthquake by exploiting the pickings of the first arrivals of the P and S waves obtained from the waveforms meeasured by a set of seismic station in the surrouning of the earthquake. In particular, our goal is to estimate the origin time, longitude, latitude and depth of the earthquake and to estimate the uncertainty of these estimates as well. 
# 
# In doing it, we will need a velocity model in order to modelling the several paths of the P and S waves from the hypocenter to the seismic stations and quantify the travel times. In particular, we will adopt the same velocity model used by the Istituto Nazionale di Geofisica e Vulcanologia for locating most of the earthquakes occurred in Italy. It is composed by the upper and lower crusts and the lithospheric mantle. The Conrad (between the upper and lower crusts) and Moho (between the lower crust and the lithospheric mantle) are set at 11 and 26.9 km depth.
# 
# ```
# 

# ## Inverse problem theory
# 
# ```{div} full-width
# 
# We now rewiev the minimal aspects ot the inverse problem thoery that we need for estimating the origin time and spatial coordinates of an earthquake. More details about this topic can be found in {numref}`Appendix %s - Inverse problem theory <chap-I>`, including the difference between volumetric and density probability functions and how taking into account modelling errors.  
# 
# ```
# 
# ### Observation equation
# 
# ```{div} full-width
# 
# Let us write the so called observation equation
# 
# $$
# \begin{align}
# & y = f(m) + \varepsilon 
# \end{align}
# $$ (E:1)
# 
# where $f$ is the function expressing the theoretical relationship between the model parameters $m$ and the observations $y$, and $\varepsilon$ are the random errors that we assume to obey to a (N-dimensional) Gaussian distribution with zero mean and covariance $\mathrm{C}$
# 
# $$
# \begin{align}
# & p(\varepsilon) = \frac{1}{\sqrt{(2\,\pi)^N\,\det(C)}}\,\exp\left(-\frac{\varepsilon^T\,C^{-1}\,\varepsilon}{2}\right) 
# \end{align}
# $$ (E:2)
# 
# In particular, $y$, $\varepsilon$ and $m$ are the arrays collecting the $N$ observations $y_i$, the $N$ random errors $\varepsilon_i$ and the $M$ model parameters $m_i$
# 
# $$
# \begin{align}
# & y = \big(y_1,\cdots,y_N\big) \qquad \varepsilon = \big(\varepsilon_1,\cdots,\varepsilon_N\big) \qquad m =(m_1,\cdots,m_M)
# \end{align}
# $$ (E:3)
# 
# Just because $\varepsilon=y-f(m)$, we can see eq. {eq}`E:2` as the conditional probability denstity function (PDF) for the observations given the model parameters 
# 
# $$
# \begin{align}
# & p(y|m) = \frac{1}{\sqrt{(2\,\pi)^N\,\det(C)}}\,\exp\left(-\frac{W(m)}{2}\right) 
# \end{align}
# $$ (E:4)
# 
# with $W$ being the so called weighted residual square sum (WRSS)
# 
# $$
# \begin{align}
# & W(m) = (y-f(m))^T\,C^{-1}\,(y-f(m))
# \end{align}
# $$ (E:5)
# 
# ```
# 
# 
# <font size="4"> Homogeneous half-space model </font>
# 
# ```{div} full-width
# 
# Let us now consider the simple half-space homogeneous Earth models and denote with $c_p$ and $c_s$ being the velocities of the P and S waves, respectively, and the model parameters $m$ as the origin time $t_e$ and spatial coordinates $\mathbf{x}_e$
# 
# $$
# \begin{align}
# & m=(t_e,\mathbf{x}_e)
# \end{align}
# $$ (E:6)
# 
# In this case, the observation equation {eq}`E:1` can be further specified as follows
# 
# $$
# \begin{align}
# & y_i = f_i(m) + \varepsilon_i  \qquad
# \end{align}
# $$ (E:7)
# 
# with
# 
# $$
# \begin{align}
# & f_i(m) =  t_e + \frac{|\mathbf{x}_i-\mathbf{x}_e|}{c_i}
# \end{align}
# $$ (E:8)
# 
# Here, $y_i$ and $\varepsilon_i$ are the time and the random error of the $i$-th pickings, while $c_i$ and $\mathbf{x}_j$ are  the seismic velocity of its phase ($c_p$ or $c_s$) and the spatial coordinates of its seismic station. Furthermore, assuming that there are no modelling errors and that the observations are independent, the covariance $C$ of the random errors is a simple diagonal matrix
# 
# $$
# \begin{align}
# & C = \mathrm{diag}(\sigma_1^2,\cdots,\sigma_N^2)
# \end{align}
# $$ (E:9)
# 
# with the variances $\sigma_i^2$ as diagonal elements. Here, $\sigma_i$ is the standard deviation of the $i$-th picking. 
# 
# We note that, in this simple case, the inverse of the covariance $C $ simply reads
# 
# $$
# \begin{align}
# & C ^{-1} = \mathrm{diag}(\sigma_1^{-2},\cdots,\sigma_N^{-2})
# \end{align}
# $$ (E:10)
# 
# and the WRSS, eq. {eq}`E:5` simplifies into
# 
# $$
# \begin{align}
# & W(m) = \sum_{i=1}^N \left(\frac{y_i-f_i(m)}{\sigma_i}\right)^2
# \end{align}
# $$ (E:11)
# 
# Recasted in this form, we can understand that each term of the summation defining the WRSS is the residual square $(y_i-f_i(m))^2$ weighted by the variance $\sigma_i^2$.
# 
# ```
# 
# ### Posterior probability density function (PDF)
# 
# ```{div} full-width
# 
# The posterior PDF is the conditional PDF $p(m|y)$ of the model parameters given the observaions. In order to obtain the posterior PDF, let us now assume that the prior information on the model parameters are described by the PDF $p(m)$ and make use of it for defining the PDF for both the observations and the model parameters
# 
# $$
# \begin{align}
# & p(y,m) = p(y|m)\,p(m)
# \end{align}
# $$ (E:12)
# 
# In this respect the prior information can be seen as the marginal PDF for the model parameters 
# 
# $$
# \begin{align}
# & p(m) =\int_{\mathbb{R}^N} p(y,m)\,d y = p(m)\,\int_{\mathbb{R}^N} p(y|m)\,d y  = p(m)
# \end{align}
# $$ (E:13)
# 
# In order to obtain the posterior PDF $p(m|y)$, we rewrite eq. {eq}`E:12` with the roles of $y$ and $m$ reversed
# 
# $$
# \begin{align}
# & p(y,m) = p(m|y)\,p(y)
# \end{align}
# $$ (E:14)
# 
# with $p(y)$ being the marginal PDF for the observation
# 
# $$
# \begin{align}
# & p(y) = \int_{\mathcal{M}} p(y,m)\,dm
# \end{align}
# $$ (E:15)
# 
# where $\mathcal{M}\subseteq\mathbb{R}^M$ is the model parameter space. The latter can be a subset of the M-dimensional space just because some model parameters can be constrained to some sepcific region of it, as for instance the vertical (or radial) coordinate of the origin that we assume to be non-positive (i.e., we will consider only positive depth, within the solid Earth).
# 
# From the comparison between eqs. {eq}`E:12` and {eq}`E:14`, we thus obtain the so called Bayes' theorem
# 
# $$
# \begin{align}
# & p(m|y) = \frac{p(y|m)\,p(m)}{p(y)}
# \end{align}
# $$ (E:16)
# 
# We note that the denominator $p(y)$ of the right-hand side of eq. {eq}`E:16` is needed only for normalizing the posterior PDF and, so, we can write
# 
# $$
# \begin{align}
# & p(m|y) \propto p(y|m)\,p(m)
# \end{align}
# $$ (E:17)
# 
# Further assuming that we do not have prior information and that the prior PDF is constant, the Bayes' theorem simply states that the posterior PDF is simply proportional to the conditional PDF of the observations given the model parameters
# 
# $$
# \begin{align}
# & p(m|y) \propto p(y|m)
# \end{align}
# $$ (E:18)
# 
# ```
# 
# ### Additional weights
# 
# ```{div} full-width
# 
# As we will see, the pickings from the Istituto Nazionale di Geofisica e Vulcanologia are provided with both the uncertainties (i.e., their standard derivations $\sigma_i$) and additional weights $w_i$. This means that the WRSS, eq. {eq}`E:11` is actualy modified as follows
# 
# $$ 
# \begin{align}
# & \displaystyle
# W(m) = \sum_{i=1}^N w_i\,\left(\frac{y_i-f_i(m)}{\sigma_i}\right)^2 
# \end{align}
# $$ (E:19)
# 
# where each weighted residual square is multiplied by the additional weight $w_i$. The use of additional weight is quite common in the earthquake localization because it allows to give more weights to the early pickings (i.e., to the pickings of those seismic stations closer to the earthquake origin). It easy to understand that their introduction is equivalent to scale the original standard deviations $\sigma_i$ by $w_i^{-1/2}$
# 
# $$ 
# \begin{align}
# & \displaystyle
# W(m) = \sum_{i=1}^N \left(\frac{y_i-f_i(m)}{\sigma_i\,w_i^{-1/2}}\right)^2 
# \end{align}
# $$ (E:20)
# 
# In light of this and of eqs. {eq}`E:4` and {eq}`E:18`, we thus obtain
# 
# $$
# \begin{align}
# & p(m|y) = k\,\exp\left(-\frac{W(m)}{2}\right)
# \end{align}
# $$
# 
# where $k$ is a normalization constant that we will not need to determine. 
# 
# Neglecting the differences between volumetric and density probability functions discussed in {numref}`Appendix %s - Inverse problem theory <chap-I>`, we can estimate the best (or most likelihood) model parameters by minimizing thw WRSS. In other words, the model parameteres $m^*$ that minimize $W$ will be our estimate of the origin time and spatial coordinates of the earthquake
# 
# $$
# \begin{align}
# & W(m^*) = \min_{m\in\mathcal{M}} W(m)
# \end{align}
# $$
# 
# ```

# ## Data set
# 
# In this section we will obtain all the information that we need to define the inverse problem, which means that we will obtain the picking times and all the additional information that we need for implementing the theoretical relationship between the model parameters and the observations, as well as calculating the WRSS.
# 
# 
# ### Earthquake selection
# 
# ```{div} full-width
# 
# Let us start by importing the main python libraries
# 
# ```

# In[1]:


get_ipython().run_line_magic('matplotlib', 'widget')
import lab
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
from myst_nb import glue
np.set_printoptions(precision=3,suppress=True)
import warnings
warnings.filterwarnings("ignore")


# ```{div} full-width
# 
# and reading the seismic inventory and the catalog containing only the 2016-08-24 M6 Accumuli earthquake that we downloaded in the  {numref}`Lecture %s <chap-L1>` {ref}`chap-L1`
# 
# ```

# In[3]:


from obspy.core.event import read_events
from obspy.core.inventory import read_inventory

directory = "Amatrice-Norcia-Visso/"

file = directory+"Seismic_Inventory.xml"
inventory = read_inventory(file)

file = directory + "7073641_Accumuli.xml"
catalog = read_events(file)


# ```{div} full-width
# 
# and we select the only event in the catalog 
# 
# ```

# In[49]:


event = catalog[0]
print(event)


# ```{div} full-width
# 
# Giving a look to the picks, we can understand that each `pick` contains information like the waveform from which has been evalueted, the phase of seismic (P or S) waves considered, the time (in `UTCDateTime`) of its arrival at the seismic station and the author who performed the measurement
# 
# ```

# In[50]:


for pick in event.picks:
    print(pick)


# ```{div} full-width
# 
# Also, checking the preferred `origin` of the event, we can see that now there is an additional field named `arrivals`
# 
# ```

# In[51]:


origin = event.preferred_origin()
print(origin)


# ```{div} full-width
# 
# Giving a look to the list `origin.arrivals`, we can understand that each `arrival` contains information like the `pick_id` (which can be used for associated the specific `arrival` to one `pick` of the list of the `event.picks`), the epicentral `distance` (in degrees) from the estimated origin to the seismic station, the `time_residual` of the observed and modelled travel time ànd the `time_weight` (in percent, so we shall divide them by 100) that we already discussed in section ()
# 
# ```

# In[52]:


for arrival in origin.arrivals:
    print(arrival)


# ```{div} full-width
# 
# Checking the number of picks and of arrivals, we can see that only a fraction of the former (about one third) have been actually used for locating the earthquake
# 
# ```

# In[53]:


print("number of picks:    ",len(event.picks))
print("number of arrivals: ",len(origin.arrivals))


# ```{div} full-width
# 
# This is because the list `event.picks` contains all the picking made by all the authors that studied this earthquakes, while the list `origin.arrivals` contains only the arrivals used for estimating the preferred `origin`.  Also, as we can see, each `arrival` have a field `pick_id` that can be used to associate it to a specific `pick`. We will exploit this fact in the next section.
#  
# 
# In addition to it, plotting the time weights of the arrivals against the epicentral distance, we can see that the assigned weight decrease with the epicentral distance and that only arrivals within the first 100 km have been actually used (the weights at greater distances being zeros). Also, we can see that all the arrivals for the S waves are weighted by about one half with respect to the arrivals for P waves at similar distances. This is beacuse the picking of the S waves is much more questionable than that of the P waves.
# 
# ```

# In[54]:


earth_radius = 6371.009
deg2km = np.pi/180 * earth_radius

fig,ax = plt.subplots(tight_layout=True)
for phase in ["P","S"]:
    distances = [ arrival.distance * deg2km for arrival in origin.arrivals if arrival.phase == phase]
    weights   = [ arrival.time_weight / 100 for arrival in origin.arrivals if arrival.phase == phase]
    ax.scatter(distances,weights,label=phase,marker=".")
ax.set_xlabel("Distance [km]")
ax.set_ylabel("Weight")
ax.legend();


# ### Picking selection

# Let us thus select only those picks that have been used in the earthquake localization of the preferred origin

# In[55]:


pick_ids = [arrival.pick_id for arrival in origin.arrivals]
origin_picks = [pick for pick in event.picks if pick.resource_id in pick_ids]

print("Number of picks:                            ",len(event.picks))
print("Number of arrivals:                         ",len(origin.arrivals))
print("Number of picks associated to the arrivals: ",len(origin_picks))


# ```{div} full-width
# 
# As we can see, we succeed in selecting only the picks associated to the arrivals.
# 
# Let us now obtain and organize all the informations that we need for defining the inverse problem of the earthquake location with similar settings to those used by the Istituto Nazionale di Geofisica e vulcanologia. First we make a lists of picks and arrivals where, for each arrival with a non-zero weight we associated the corresponding pick.
# 
# ```

# In[12]:


picks_arrivals = []
for arrival in origin.arrivals:
    if arrival.time_weight == 0.0: continue
    for pick in origin_picks:
        if pick.resource_id == arrival.pick_id:
            picks_arrivals.append([pick,arrival])
            break
print("Number of selected picks and arrivals:",len(picks_arrivals))


# ```{div} full-width
# 
# From this list (which contains only 56 couples of picks and arrivals, because all the others have zero weight) we can obtain the piciking time and its uncertainity (or standard deviations) from the pick and the assigned weight from the arrival. In addition to it, we shall obtain the spatial coordinate of the seismic station (longitude, latitude and elevation) to which each picking refers. Unfortunatly, the geopgarphical coordinate are not ready available niether from the pick or the arrival and, so, we shall make use of the waveform identifiers contained in the picks along with an inventory of the seismic networks from which we can obtain the geographical coordinates. In this perspective, we download all the seismic stations within about 100 km (1 degrees) from the epicenter that have operate one hour before and one hour after the origin time.
# 
# ```

# In[56]:


circle = dict(maxradius=1,longitude=origin.longitude,latitude=origin.latitude)


# ```{div} full-width
# 
# and plot it along with the catalog containing the only Accumoli earthquake
# 
# ```

# In[57]:


fig = lab.setup_map(circle,color="red")
fig = inventory.plot(fig=fig,label=False,size=30,show=False,color="orange")


# ```{div} full-width
# 
# Then, we extent the list of picks and arrivals including also the channels from the inventory corresponding to those to which the picks refer
# 
# ```

# In[58]:


picks_arrivals_channels = []
for pick, arrival in picks_arrivals:
    waveform_id = pick.waveform_id.id
    codes = waveform_id.split(".")
    inv = inventory.select(*codes)
    try:
        channel = inv[0][0][0]
        picks_arrivals_channels.append([pick,arrival,channel])
    except:
        print("Warning! There is no the channel "+waveform_id+" to which the pick refer")


# ```{div} full-width
# 
# The `try` and `except` syntax has been used to prevent from an error in the case in which the channel is not found in the inventory. Fortunately, in the present case, all the channels that we need have been found.
# 
# In order to have an idea of how many seismic stations have been considered by the INGV for locating the earthquake, we now make the same plot as before but indicating with the green mark (and the station code) those seismic stations that have been actually used.
# 
# ```

# In[59]:


import cartopy

fig = lab.setup_map(circle,color="red")
fig = inventory.plot(fig=fig,label=False,size=30,show=False,color="orange")

ax = fig.axes[0]
for pick, arrival, channel in picks_arrivals_channels:
    label = pick.waveform_id.id.split(".")[1]
    ax.scatter(channel.longitude,channel.latitude,marker="v",color="green",transform=cartopy.crs.PlateCarree(),zorder=30)
    ax.annotate(label,xy=(channel.longitude+0.02,channel.latitude),transform=cartopy.crs.PlateCarree(),zorder=30)


# ```{div} full-width
# 
# Among all these channels, it will be useful to identify the closest one, i.e., the one for which has been made the early picking
# 
# ```

# In[60]:


times = [ pick.time for pick, _, _ in picks_arrivals_channels ]

i0 = np.argmin(times)
ref_pick, ref_arrival, ref_channel = picks_arrivals_channels[i0]

ref_time = ref_pick.time
ref_longitude, ref_latitude = ref_channel.longitude, ref_channel.latitude
ref_code = ref_pick.waveform_id.id

print("ref_code: ",ref_code)
print("ref_time:",ref_time)
print(ref_longitude,ref_latitude)


# In[61]:


from myst_nb import glue
glue("ref_code", ref_code)


# As far as we know, without performing the full inversion of the observations, the picking time, longitude and latitude of the channels {glue:}`ref_code` is the  

# ### Data organization and the azimuthal equidistant projection

# ```{div} full-width
# 
# We are now in the position of obtaining all the informations that we need for defining the inverse problem: the picking time, the longitude, latitude and elevation of the seismic station, the picking uncertainity, the weight assigned to the arrival and the phase. 
# 
# Rather than obtaining the picking time in UTCDateTime, we will consider a reference time and compute the differences with respect to it in order to obtain the picking time in seconds
# 
# ```

# In[62]:


N = len(picks_arrivals_channels)
data = np.empty((N,6))
for i,(pick,arrival,channel) in enumerate(picks_arrivals_channels):
        
        T = pick.time - ref_time
        E = pick.time_errors["uncertainty"]
        scale = arrival.time_weight/100
        E /= np.sqrt(scale)

        lon,lat = channel.longitude, channel.latitude
        Z = channel.elevation
        
        phase = int(pick.phase_hint == "S")

        data[i] = [T,lon,lat,Z,E,phase]
        
print("{0:>9}  {1:>9}  {2:>9}  {3:>9}  {4:>9}  {5:>9}".format("t [s]","lat [°]","lon [°]","z [m]","std [s]","phase"))
for datum in data:
    print("{0:>9.4f}  {1:>9.4f}  {2:>9.4f}  {3:>9.4f}  {4:>9.4f}  {5:>9d}".format(*datum[:-1],int(datum[-1])))


# ```{div} full-width
# 
# In order to simplify the modelling of the travel times of the seismic wave to the seismic station and because we are considering a very small area of the Earth surface, we will consider a Cartesian reference system and use the elevation as vertical coordinate $x_3$ and the $x_1$ and $x_2$ coordinates of the azimuthal equidistant projection for the horizontal coordinates. The azimuthal equidistant projection is implemented in the module `cartopy`. As already discussed in {numref}`Lecture %s <chap-L1>` {ref}`chap-geographic-maps`, here the syntax required for the coordinate change from the geographical reference system to the Cartesian one
# 
# ```

# In[63]:


projection = cartopy.crs.AzimuthalEquidistant(central_longitude=ref_longitude, central_latitude=ref_latitude)
geodetic   = cartopy.crs.Geodetic()

llz = data[:,1:4].T  #longitude, latitude, elevation
xyz = projection.transform_points(geodetic,*llz)
xyz /= 1e3

data[:,1:4] = xyz #Cartesian coordinates x, y, z

print("{0:>9}  {1:>9}  {2:>9}  {3:>9}  {4:>9}  {5:>9}".format("t [s]","x [km]","y [km]","z [km]","std [s]","phase"))
for datum in data:
    print("{0:>9.4f}  {1:>9.4f}  {2:>9.4f}  {3:>9.4f}  {4:>9.4f}  {5:>9d}".format(*datum[:-1],int(datum[-1])))


# In[65]:


ingv = np.array([origin.time-ref_time,origin.longitude,origin.latitude,-origin.depth]) # [s,deg,deg,m]
llz = ingv[1:]  #longitude, latitude, elevation
xyz = projection.transform_points(geodetic,*llz)
xyz /= 1e3
ingv[1:] = xyz #Cartesian coordinates x, y, z


deg2km = np.pi/180 * earth_radius

ingv_std = np.empty(4)
ingv_std[0] = origin.time_errors["uncertainty"]
rlat = origin.latitude * np.pi/180
ingv_std[1] = origin.longitude_errors["uncertainty"] * deg2km * np.cos(rlat)
ingv_std[2] = origin.latitude_errors["uncertainty"]  * deg2km
ingv_std[3] = origin.depth_errors["uncertainty"] / 1e3

keys = ["T0 [s]","X0 [km]","Y0 [km]","Z0 [km]"]

print("\n{0:>8} {1:>8} {2:>8} ".format("","val","std"))
for key,val,std in zip(keys,ingv,ingv_std):
    print("{0:<8} {1:>8.4f} {2:>8.4f}".format(key,val,std))


# ```{div} full-width
# 
# In the list of arrivals it is also contained the time residual of each arrival, that is the difference between the picking time and the one modelled with the velocity model of the INGV. We use it to obtain the modelled times by INGV and we plot the data with respect to the origin and compare them with respect to the modelled ones
# 
# ```

# In[23]:


residues = np.array([arrival.time_residual for _, arrival, _ in picks_arrivals_channels])
modelled_times = data[0]- residues 
fig = lab.plot_data(data,origin=ingv,modelled_times=modelled_times)


# ```{div} full-width
# 
# Let us now make the same figure plotting also two straight lines according to a simple velocity model with $v_p = 5$ km/s and $v_s = 3$ km/s
# 
# ```

# In[24]:


fig = lab.plot_data(data,origin=ingv,modelled_times=modelled_times)
T0,T1 = ingv[0],15
fig.axes[0].plot([T0,T1],[0,5*(T1-T0)],color="gold");
fig.axes[0].plot([T0,T1],[0,3*(T1-T0)],color="gold");


# ## Homogeneous half-space velocity model
# 
# ```{div} full-width
# 
# In the simple case of an homogeneous half-space velocity model, we only need to specify the P and S wave seismic velocity
# 
# ```

# In[52]:


velocity_model = np.array([6,3])
print("velocity model:",velocity_model)


# ```{div} full-width
# 
# Let us define a function which compute the travel times for a set of hypocentral distances `Ds`, taking info account for the seismic phases `Ps` and the specified velocity model. This function correspons to the thoeoretical relationship between the model parameters and the observations, eq. {eq}`E:8`.
# 
# ```

# In[53]:


def eva_travel_times(Ds,Ps,velocity_model):
    
    VP,VS = velocity_model
        
    DTs_mod = np.empty(len(Ds))
    for i,(D,P) in enumerate(zip(Ds,Ps)):
        if P == 1:
            DTs_mod[i] = D/VS
        else:
            DTs_mod[i] = D/VP

    return DTs_mod


# ```{div} full-width
# 
# Then, we use it for defining another function which compute the weighte residual square sum (WRSS)), eq. {eq}`E:19`, for the model parameters $m$ collecting the origin time `TE` and cartesian coordinates `XE`, `YE` and `ZE` once given the `data` and the `velocity_model`
# 
# ```

# In[54]:


def eva_wrss(m,data,velocity_model):
    
    TE,XE,YE,ZE = m
    Ts,Xs,Ys,Zs,Es,Ps = data
    Ds = np.sqrt( (Xs-XE)**2 + (Ys-YE)**2 + (Zs-ZE)**2 )

    Ts_mod = eva_travel_times(Ds,Ps,velocity_model) + TE

    WRSS = (((Ts-Ts_mod)/Es)**2).sum()
        
    return WRSS


# ```{div} full-width
# 
# Here the so calculated WRSS for the origin `ingv` by the INGV according to the homogeneous half-space velocity model
# 
# ```

# In[55]:


wrss = eva_wrss(ingv,data,velocity_model)
print("wrss for the ingv:",wrss)


# ```{div} full-width
# 
# Making use of the submodule `optimize` from `scipy`, we can define the model space using the class `optimize.Bounds` and minimize the WRSS as function of the model parameters using the function `optimize.minimize`. By default, in presence of `bounds, the minimization is made according to the L-BFGS-B method described in [Byrd et al. (1995)](https://epubs.siam.org/doi/10.1137/0916069) {download}`<../_static/Byrd_etal_1995.pdf>` and [Zhu et al. (1997)](https://dl.acm.org/doi/10.1145/279232.279236).
# 
# Byrd, R H and P Lu and J. Nocedal. 1995. A Limited Memory Algorithm for Bound Constrained Optimization. *SIAM Journal on Scientific and Statistical Computing* 16 (5): 1190-1208.
# [7]
# 
# Zhu, C and R H Byrd and J Nocedal. 1997. L-BFGS-B: Algorithm 778: L-BFGS-B, FORTRAN routines for large scale bound constrained optimization. *ACM Transactions on Mathematical Software* 23 (4): 550-560.
# ```

# In[56]:


from scipy import optimize

ub = [np.inf,np.inf,np.inf,0]
bounds = optimize.Bounds(ub=ub)  # definition of the upper bounds for the model parameters (i.e., the vertical coordinate ZE must be non-positive

fun = lambda m: eva_wrss(m,data,velocity_model)   # function to be minimized 

guess = [0,0,0,-10]   # starting point for the minimization
sol = optimize.minimize(fun, guess, bounds=bounds)     # minimizationb
print("Minimization solution:\n\n",sol)


# ```{div} full-width
# 
# We note that we have made use of the `lambda` syntax for defining the function `fun` of the only model parameters starting from the python function `eva_wrss`. Also, the function `optimize.minimize` return the object  `sol` which summarize the main information about the mninimization, including where the function takes its minimum, at `sol.x`, and the minimum value `sol.fun`.
# 
# In order to verify that we found a minimum, let us plot the WRSS by varying one model parameter at the time and fixing the other to their most likelihood values
# 
# ```

# In[57]:


def plot_wrss(mb, exts, fun):

    fig,axes = plt.subplots(2,2,tight_layout=True,figsize=(8,8))
    axes = axes.flatten()
    for k,(key,ext) in enumerate(exts.items()):
        ax = axes[k]
        m = mb.copy()
        xs = np.linspace(*ext,1000) + mb[k]
        fs = np.empty(len(xs))
        for i,x in enumerate(xs):
            m[k] = x
            WRSS = fun(m)
            fs[i] = WRSS
        ax.plot(xs,fs)
        ax.axvline(mb[k],linewidth=0.5)
        ax.axhline(fs.min(),linewidth=0.5)
        ax.set_xlabel(key)
        ax.set_ylabel("WRSS")
        
    wrss = fun(mb)
    text = "WRSS: {0:>10.3f}    Model parameters: "+" ".join([ "{"+str(k)+":>7.3f}" for k in range(1,5) ])
    fig.suptitle(text.format(wrss,*mb))
    
    return fig


# In[58]:


exts = { "TE [s]":[-0.2,0.2], "XE [km]": [-1,1], "YE [km]": [-1,1], "ZE [km]":[-1,1] }
fig = plot_wrss(sol.x, exts, fun)


# ## Two layered half-space velocity model
# 
# Let us imporve the velocity model by considering a two-layered Earth model consisting of the the upper and lower crusts separated by the Conrad discontinuity at $11$ km depth. 
# 
# In doing this, let us consider a Cartesian reference system and denote with $\ell$ and $z$ the horizontal and vertical cordinates and set the hypocenter at $(0,z_e)$ and the seismic station at $(\ell,z)$, with $z_e\leq 0$ and $z\geq 0$, as depicted in {numref}`Figure {number} <fig-velocity-model>`.
# 
# 
# ```{figure} ../images/Velocity_model.png
# ---
# align: center
# figclass: margin-caption
# name: fig-velocity-model
# ---
# Cartoon depicting the Cartesian reference system and the ray path from the hypocenter at $(0,z_e)$ to the seismic station at $(\ell,z)$. The dashed line indicates the Conrad discontinuity at $H=11$ km depth. We assume that the vertical coordinate of the seismic station is non-negative, $z\geq 0$, and denote with $\ell_c$ the horizontal component of the ray path within the lower crust.
# ```
# 
# <font size="4"> Hypocenter in the upper crust </font>
# 
# When the hypocenter is in the upper crust ($z_e \geq -H$), we have to compare the direct (red line) and headwave (green line) travel times $t_d$ and $t_h$ and select the smaller one
# 
# $$
# t = \min(t_d,t_h)
# $$ (LE:1)
# 
# with 
# 
# $$
# t_d = \frac{\sqrt{\big(z-z_e\big)^2 + \ell^2}}{c_0}
# \qquad\mathrm{and}\qquad
# t_h = \frac{\ell_c}{c_1} + \frac{ \sqrt{\Delta z^2 + \Delta \ell^2}}{c_1}
# $$ (LE:2)
# 
# Here, $\Delta \ell$ and $\Delta z$ are the horizontal and vertical components of the ray path of the headwave in the upper crust
# 
# $$
# \Delta l = \ell-\ell_c \geq 0 \qquad\mathrm{and}\qquad
# \Delta z = (H-z_e) + (z+H)  = 2\,H + z - z_e
# $$
# 
# In order to determine $\ell_c$, we note that the incident angle $\theta$ is given by
# 
# $$
# \tan\theta = \frac{\Delta \ell}{\Delta z}
# $$
# 
# and that the refraction angle (in the lower crust) must be $\pi/2$. In this respect, from the Snell's law, the indident angle must be
# 
# $$
# \sin\theta = \frac{c_0}{c_1}  \qquad\Rightarrow\qquad \tan\theta = \frac{\sin\theta}{\cos\theta} = \frac{c_0}{\sqrt{c_1^2-c_0^2}}
# $$
# 
# We thus obtain 
# 
# $$
# \Delta \ell = \tan\theta\,\Delta z\qquad\Rightarrow\qquad
# \ell_c = \ell - \Delta\ell = \ell - \tan\theta\,\Delta z
# $$
# 
# and we can rewrite the headwave travel time as follows
# 
# $$
# t_h = \frac{\ell_c}{v_1} + \Delta \ell\,\sqrt{\frac{c_1^2}{c_0^2}-1}
# $$
# 
# We note that the headwave is possible only when $\Delta \ell \geq 0$ or, equivalently, $\ell\geq \ell_c$.
# 
# <font size="4"> Hypocenter in the lower crust </font>
# 
# When the hypocenter depth is the lower crust ($z_e<-H$), the travel time $t$ depend on the horizontal coordinate $x$ where the ray path encounter the Conrad discontinuity
# 
# $$
# t(\ell_c) = \frac{ \sqrt{\big(H+z_e\big)^2+\ell_c^2}}{v_1} + \frac{ \sqrt{(H+z)^2+\big(\ell-\ell_c\big)^2} }{v_0}
# $$ (EL2:1)
# 
# 
# 
# In order to determine $\ell_c$ we have to minimize eq. {eq}`EL2:1` within the interval  $[0,L]$
# 
# $$
# \frac{d t(\ell_c)}{d\ell_c} = \frac{x}{v_1\,\sqrt{(H-z_e)^2+x^2}} - \frac{\ell-\ell_c}{v_0\,\sqrt{(z-z_e)^2+(\ell-\ell_c)^2}} = 0
# $$
# 
# By equating the square of the two terms in the right-hand side and multuplying by the minimum common denominator, eq. () becomes
# 
# $$
# x^2\,v_0^2\,(z-z_e)^2+x^2\,\big(v_0^2-v_1^2\big)\,(L-x)^2 - (L-x)^2\,v_1^2\,(H-z_e)^2 = 0
# $$
# 
# By introducing the following normalization
# 
# $$
# \ell_c = L\,\hat{\ell_c} 
# $$
# 
# and defining
# 
# $$
# \begin{align}
# & A_0 = \frac{v_0^2}{v_1^2-v_0^2}\,\frac{(z-z_e)^2}{\ell^2} = \frac{S\,B_0}{1-S} & \quad  A_1 = \frac{v_1^2}{v_1^2-v_0^2}\,\frac{(h-z_e)^2}{\ell^2} = \frac{B_1}{1-S}
# \end{align}
# $$
# 
# with
# 
# $$
# \begin{align}
# & S = \frac{v_0^2}{v_1^2} & \quad B_0 = \frac{(z-z_e)^2}{\ell^2}  \qquad B_1 = \frac{(h-z_e)^2}{\ell^2}
# \end{align}
# $$
# 
# we thus obtain that $\hat{\ell}\in[0,1]$ must be a root of the following 4-order polynomial
# 
# $$
# \hat{\ell}^4\,-2\,\hat{\ell}^3  + \hat{\ell}^2\,(1 + A_1-A_0) - 2\,\hat{\ell}\,A_1 + A_1 = 0
# $$
# 
# By definition, there is only one real root of te above polynomial in the normalized interval $[0,1]$.
# 

# ### Optimization
# 
# Here, we define the two layered velocity model

# In[59]:


velocity_model = [5.00, 6.50, 2.89, 3.75, -11.1 ]


# and we implement the python function computing the modelled travel time of the two-layered Earth model 

# In[60]:


#################################################################
def eva_travel_time(L, Z, ZE, V0, V1, H):
    
    sin0 = V0/V1
    S = sin0**2
    
    if ZE >= H:
        
        D = np.sqrt(L**2+(Z-ZE)**2)
        T = D/V0
        
        if sin0 < 1:
        
            cos0 = np.sqrt(1-S)

            DZ  = Z + ZE - 2*H 

            D0 = DZ/cos0
            DX = D0*sin0

            if DX < L:

                T_headwave = (L-DX)/V1  +  D0/V0
                T = min(T,T_headwave)
        
    else: 
        
        if V0 == V1:
            
            D = np.sqrt(L**2+(Z-ZE)**2)
            T = D/V0

        elif L == 0:

            T = (Z-H)/V0 + (H-ZE)/V1
            
        else:

            B0 = ( (Z-H)/L )**2
            B1 = ( (H-ZE)/L )**2
            A0 = B0 * S / (1-S)
            A1 = B1 / (1-S)

            poly_coefs = [1,-2,1+A1-A0,-2*A1,A1]
            
            roots = np.roots(poly_coefs)
            real_roots = roots[roots.imag == 0].real
            
            X = real_roots[np.logical_and(real_roots >= 0,real_roots<=1)][0]
            
            T = L * ( np.sqrt(B1+X**2)/V1 + np.sqrt(B0+(1-X)**2)/V0 )
        
    return T
#################################################################


#################################################################
def eva_travel_times(Ls,Zs,Ps,ZE,velocity_model):
    
    VP0,VP1,VS0,VS1,H = velocity_model

    DTs_mod = np.empty(len(Ls))        
    for i,(L,Z,P) in enumerate(zip(Ls,Zs,Ps)):
        if P == 1:
            DTs_mod[i] = eva_travel_time(L,Z,ZE,VS0,VS1,H)
        else:
            DTs_mod[i] = eva_travel_time(L,Z,ZE,VP0,VP1,H)
        
    return DTs_mod
#################################################################


# Before solving the inverse problem of the earthquake localization, let us plot the travel times as function of the hypocentral distance for the cases $z_e=-7$ and $-13$ km (above and below the Conrad discontinuity)

# In[61]:


fig,axes = plt.subplots(1,3,tight_layout=True,figsize=(10,5))

for ZE in [-7,-13]:

    label = str(-ZE)+" [km]"
    
    Ls = np.linspace(0,100,1000)
    Zs = np.zeros(Ls.shape)
    Ps = np.zeros(Ls.shape)

    Ds1 = np.sqrt(Ls**2+(Zs-ZE)**2)
    Ts1 = eva_travel_times(Ls,Zs,Ps,ZE,velocity_model)

    Ls = 10**np.linspace(0,4,1000)
    
    Ds2 = np.sqrt(Ls**2+(Zs-ZE)**2)
    Ts2 = eva_travel_times(Ls,Zs,Ps,ZE,velocity_model)

    ax = axes[0]
    ax.plot(Ts1,Ds1,label=label)

    axes[1].plot(Ds1/Ts1,Ds1,label=label)
    axes[2].semilogy(Ds2/Ts2,Ds2,label=label)

axes[0].set_ylabel("Hypocentral distance [km]")
axes[0].set_xlabel("Travel time [s]")
axes[1].set_xlabel("Distance / Time [km/s]")        
axes[2].set_xlabel("Distance / Time [km/s]")        

for ax in axes:
    ax.axhline(0,linewidth=0.5)
    ax.legend()


# As we can see ...
# 
# Let us now make ude of the function `eva_travel_times` for compouting the WRSS

# In[69]:


def eva_wrss(m,data,velocity_model,logy=False):
    
    TE,XE,YE,ZE = m
    Ts,Xs,Ys,Zs,Es,Ps = data
    Ls = np.sqrt( (Xs-XE)**2 + (Ys-YE)**2 )

    Ts_mod = eva_travel_times(Ls,Zs,Ps,ZE,velocity_model) + TE

    WRSS = (((Ts-Ts_mod)/Es)**2).sum()
        
    if logy:
        return WRSS, Ts_mod
    else:
        return WRSS


# In[70]:


wrss = eva_wrss(ingv,data,velocity_model)
print("wrss:",wrss)


# In[71]:


wrss, Ts_mod = eva_wrss(ingv,data,velocity_model,True)


# In[72]:


fig = lab.plot_data(data,ingv,Ts_mod)


# Let us now consider the starting point that we already use for the homogeneous half-space model and find the minimum of the WRSS

# In[74]:


fun = lambda m: eva_wrss(m,data,velocity_model)
sol = optimize.minimize(fun,guess,bounds=bounds)

print(sol)


# In[75]:


exts = { "TE":[-0.2,0.2], "XE":[-1,1], "YE":[-1,1], "ZE":[-1,1] }
fig = plot_wrss(sol.x, exts, fun)


# ### Searching for multiple local minima

# In[77]:


def make_guesses(nwalkers,lims):

    spans = np.diff(lims,axis=1)
    print(spans[:,0])
    print(lims[:,0])
    ndim = len(lims)
    guesses = np.random.rand(nwalkers, ndim) * spans[:,0] + lims[:,0]

    return guesses

nwalkers = 50
mlims = np.array([[-5,0],[-20,20],[-20,20],[-20,0]])

guesses = make_guesses(nwalkers,mlims)
print("\nmean =",guesses.mean(axis=0))
print("std  =",guesses.std(axis=0))
print(guesses)


# In[78]:


sols = []
for i,guess in enumerate(guesses):
    sol = optimize.minimize(fun,guess,bounds=bounds)
    print(i,guess,sol.x,sol.fun)
    sols.append(sol)


# In[79]:


delta_max = 1e-3

uniques = []
for i,sol_new in enumerate(sols):
    logy = True
    for j in uniques:
        sol = sols[j]
        delta = np.sqrt(( (sol.x-sol_new.x)**2 ).sum())
        if delta < delta_max:
            logy = False
    if logy: 
        uniques.append(i)
        print("best =",sol.x,"fun =",sol.fun)
        
uniques = np.array(uniques)
print("uniques =",uniques)

js = np.argsort([sols[j].fun for j in uniques])
uniques = uniques[js]
print("sorted uniques =",uniques)

sols = [sols[j] for j in uniques]
for sol in sols:
    fig = plot_wrss(sol.x, exts, fun)


# ### Estimates of the uncercainties

# In[82]:


###########################################################s
def eva_functional(fun,m,N):
    
    wrss = fun(m)
    L = N * np.log(wrss)
    
    return L
###########################################################s

from scipy import linalg

###########################################################s
def make_inverse_symmetric(hess):

    w,o = linalg.eigh(hess)
    inv = (o / w) @ o.T
    return inv
###########################################################s


###########################################################s
def make_hess(functional,mb,steps):

    M = len(mb)
    H = np.empty((M,M))
    f0 = functional(mb)
    
    dm = np.eye(M) * steps
    for i in range(M):
        eps = steps[i]
        f_up = functional(mb+dm[i]) 
        f_dw = functional(mb-dm[i])
        H[i,i] = ( f_up + f_dw - 2*f0 ) / steps[i]**2
        for j in range(i):
            up =  ( functional(mb+dm[i]/2+dm[j]/2) - functional(mb-dm[i]/2+dm[j]/2) ) / steps[i]
            dw =  ( functional(mb+dm[i]/2-dm[j]/2) - functional(mb-dm[i]/2-dm[j]/2) ) / steps[i]
            H[i,j] = (up-dw) / steps[j]
            H[j,i] = H[i,j]
        
    return H
###########################################################


# In[83]:


functional = lambda m: eva_functional(fun,m,N)

steps = np.ones(M)*1e-4

ms = np.empty((len(sols),M))
Ws = np.empty(len(sols))
Ls = np.empty(len(sols))
covs = np.empty((len(sols),M,M))
for i,sol in enumerate(sols):
    m = sol.x
    ms[i] = sol.x
    Ws[i] = sol.fun
    Ls[i] = functional(m)
    H = make_hess(functional,m,steps)
    covs[i] = 2*make_inverse_symmetric(H)
    print(Ws[i])
    print(ms[i])
    print(np.sqrt(np.diag(covs[i])))
DL = Ls[0]-Ls[1]
print(DL)


# <p style="page-break-after:always;"></p>
