#!/usr/bin/env python
# coding: utf-8

# (chap-Libraries)=
# # Libraries
# 
# ```{contents} Sections
# :local:
# :depth: 2
# ```
# 
# ## Theory
# 
# Below we just copy the content of the file {Download}`theory.py<./theory.py>`

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn



#########################################################################
J = lambda x: spherical_jn(0,x)
#########################################################################

#########################################################################
def eva_box(ts, Tb=1):
    """
    \t\n
    Return the  the box function of length `Tb` evaluated at times `ts`
    """

    box = np.ones(ts.shape)
    box[ts<0]  = 0
    box[ts>=Tb] = 0
    
    return box
#########################################################################


#########################################################################
def eva_FT_box(fs, Tb=1):
    """
    \t\n
    Return the Fourier transform of the box function of length `Tb` evaluated at frequencies `fs`
    """
    
    ws = 2*np.pi*fs
    xs = ws*Tb/2
    FT_box = Tb * np.exp(-1j*xs) * J(xs)
    
    return FT_box
#########################################################################


#########################################################################
def eva_test_ground_motion(ts,Tg=1):
    """
    \t\n
    Return the test ground motion (displacement, velocity and accelarations) defined in eq. (3.59) evaluated at times `ts`
    """
    wg = 2*np.pi/Tg
    test = np.empty((3,)+ts.shape)
    
    x = wg*ts
    x2 =2*x
    test[0] = (   np.cos(x2)/4 - np.cos(x) + 3/4 ) / wg**2
    test[1] = ( - np.sin(x2)/2 + np.sin(x)       ) / wg
    test[2] = ( - np.cos(x2)   + np.cos(x)       )
    
    test[:,ts<0] = 0
    test[:,ts>Tg] = 0
    
    return test
#########################################################################



#########################################################################
def eva_test_ground_displacement(ts, Tg=1):
    """
    \t\n
    Return the test ground displacement defined in eq. (3.59) evaluated at times `ts`
    """

    test = eva_test_ground_motion(ts, Tg=Tg)[0]
    
    return test
#########################################################################


#########################################################################
def eva_FT_test_ground_displacement(fs, Tg=1):
    """
    \t\n
    Return the Fourier transform of the test ground displacement defined in eq. (5.15) evaluated at frequencies `fs`
    """

    wg = 2*np.pi / Tg
    ws = 2*np.pi*fs
    
    x = ws*Tg/2
    xg = wg*Tg/2
    xg2 = 2*xg
    
    FT_test = np.empty(fs.shape,dtype=complex)
    
    FT_test  = J(xg2-x) * np.exp(1j*(xg2-x) ) / 8    + J(xg2+x) * np.exp(-1j*(xg2+x)) / 8
    FT_test -= J(xg-x)  * np.exp(1j*(xg-x))   / 2    + J(xg+x)  * np.exp(-1j*(xg+x))  / 2
    FT_test += J(-x)    * np.exp(-1j*x)       * 3/4
    
    FT_test *= Tg / wg**2
    
    return FT_test
#########################################################################







#########################################################################
def DFT_to_periodic(cts, fs, DFT, n, d):
    """
    \t\n
    Return the approximating periodic function defined in eq. (5.31) evaluated 
    at times `cts`. It requires the discrete frequencies `fs`, the DFT 
    coefficients `DFT`, the number of samplings `n` and the time step `d`.
    """
    o = n%2
    m = n//2
    ws = 2*np.pi*fs
    T = n*d
    
    DFT_mod = 2*DFT.copy()
    DFT_mod[0] /= 2
    if o == 0: DFT_mod[-1] /= 2 
    fun = (np.exp(np.tensordot(cts,1j*ws,axes=0)) @ DFT_mod ) .real / T
    
    return fun
#########################################################################


#########################################################################
def DFT_to_FT(cfs, fs, DFT, n, d):
    """
    \t\n
    Return the Fourier transform of the approximating function with compact 
    support defined in eq. (5.33) evaluated at frequencies `cfs`. It requires
    the discrete frequencies `fs`, the DFT coefficients `DFT`, the number of 
    samplings `n` and the time step `d`.
    """

    o = n%2
    m = n//2
    ws = 2*np.pi*fs
    cws = 2*np.pi*cfs
    T = n*d
    
    xs = ws * T/2
    cxs = cws * T/2
    
    xs2,cxs2 = np.meshgrid(xs,cxs)
    xk_p = xs2-cxs2
    xk_m = -xs2-cxs2
    
    ck = np.ones(m+1)
    ck[0] /= 2
    if o == 0: ck[-1] /=2
    
    DFT_mod = DFT * ck
    FT = ( J(xk_p) * np.exp(1j*xk_p) ) @ DFT_mod +  ( J(xk_m) * np.exp(1j*xk_m) ) @ DFT_mod.conjugate()

    return FT
#########################################################################


#########################################################################
def plot_comparison(T, d, eva_fun, eva_FT_fun=None, args=(), tmin=-1, tmax=21, fmin=0, fmax=6, marker=True):

    cts = np.linspace(tmin,tmax,100001)
    cfun = eva_fun(cts,*args)

    ts = np.arange(0,T,d)
    n = len(ts)
    fun = eva_fun(ts,*args)

    fs = np.fft.rfftfreq(n,d)
    DFT = np.fft.rfft(fun) * d
    FT = eva_FT_fun(fs,*args)

    cfun_periodic = DFT_to_periodic(cts,fs,DFT,n,d)

    
    if marker: scatter_options = dict(marker=".", color="red", zorder=30)
    hvline_options = dict(color="black", linewidth=0.5)
    
    
    fig1, axes = plt.subplots(2,tight_layout=True,figsize=(10,4))
    fig1.suptitle("Comparison in the time domain")

    ax = axes[0]
    ax.plot(cts,cfun,label="original")
    line = ax.plot(cts,cfun_periodic,label="approximation")[0]
    if marker is not None: ax.scatter(ts,fun,**scatter_options)

    ax = axes[1]
    whe = np.logical_and(cts>=0,cts<=T)
    ax.plot(cts[whe],cfun[whe]-cfun_periodic[whe],label="difference")
    if marker:  ax.scatter(ts,np.zeros(n),**scatter_options)

    for ax in axes:
        ax.legend()
        ax.axhline(0,**hvline_options)
        ax.axvspan(0,T,color="gray",alpha=0.2)
    axes[-1].set_xlabel("Time [s]")

    if eva_FT_fun is None:
        return fig1,None

    cfs = np.linspace(fmin,fmax,10001)
    cFT = eva_FT_fun(cfs,*args)
    
    cFT_periodic = DFT_to_FT(cfs,fs,DFT,n,d)

    fig2, axes = plt.subplots(3,tight_layout=True,figsize=(10,6),sharex=True)
    fig2.suptitle("Comparison in the frequency domain")

    ax = axes[0]
    ax.set_ylabel("Real part")
    ax.plot(cfs,cFT.real,label="original")
    ax.plot(cfs,cFT_periodic.real,label="approximation")
    if marker: ax.scatter(fs, DFT.real, **scatter_options)

    ax = axes[1]
    ax.set_ylabel("Imaginary part")
    ax.plot(cfs,cFT.imag,label="original")
    ax.plot(cfs,cFT_periodic.imag,label="approximation")
    if marker: ax.scatter(fs, DFT.imag, **scatter_options)

    ax = axes[2]
    ax.plot(cfs,abs(cFT-cFT_periodic),label="difference")

    for ax in axes:
        ax.axhline(0,**hvline_options)
        ax.set_xlim(fmin,fmax)

    for ax in axes:
        ax.legend()
    axes[-1].set_xlabel("Frequency [Hz]")
        
    return fig1,fig2
#########################################################################









#########################################################################
def damped_harmonic_oscillator(ts, U, V, natural_period=1, damping_ratio=0.1):
    """
    \t\n
    Return the `transient_motions` (displacement, velocity, and acceleration)
    of a damped harmonic oscillator evaluated  at the times `ts` with  
    displacement `U` and velocity `V` as initial conditions. 

    The natural period and damping ratio can be set as optional arguments.
    """
    
    w0  = 2*np.pi/natural_period
    wd  = w0 * np.sqrt(1-damping_ratio**2)
    lam = w0*damping_ratio
    
    A = - U*w0**2 - 2*V*lam
    
    transient_motions = np.empty((3,)+ts.shape)
    
    transient_motions[0] = ( U *np.cos(wd*ts) + (U*lam + V)/wd       * np.sin(wd*ts) ) *  np.exp(-lam*ts)
    transient_motions[1] = ( V *np.cos(wd*ts) - (U*w0**2 + V*lam)/wd * np.sin(wd*ts) ) *  np.exp(-lam*ts)
    transient_motions[2] = ( A *np.cos(wd*ts) - (V*w0**2 + A*lam)/wd * np.sin(wd*ts) ) *  np.exp(-lam*ts)
    
    return transient_motions
#########################################################################
    
    
#########################################################################    
def setup_harmonic_oscillator(wg, natural_period=1, damping_ratio=0.1):
    """
    \t\n
    Return the amplitude factor and the phase shift of a damped harmonic 
    oscillator for the given anuglar frequency wg. By default, the natural 
    period and dampimng ratio of the harmonic oscillator are 1 s and 0.1.
    """
    
    w0  = 2*np.pi / natural_period
    lam = w0 * damping_ratio
    
    D = np.sqrt((w0**2-wg**2)**2+(2*wg*lam)**2)
    C = (w0**2-wg**2)/D

    phase = - np.arccos(C)
    amplitude_factor = wg**2 / D
    
    return amplitude_factor, phase
#########################################################################


#########################################################################
def steadystate_driven_oscillator(ts, Tg=1, ground_phase=0, natural_period=1, damping_ratio=0.1):
    """
    \t\n
    Return ground acceleration of given ground period and phase and the
    steady state relative motion (displacement, velocity and acceleration) 
    of an harmonic oscillator evaluated at the times ts.
    """
    
    wg = 2*np.pi / Tg
    amplitude_factor, phase = setup_harmonic_oscillator(wg, natural_period, damping_ratio)
    
    x = wg * ts + ground_phase 
    ground_acceleration = np.sin(x)
    
    x += phase
    steadystate_motions = np.empty((3,)+ts.shape)
    steadystate_motions[0] = - amplitude_factor / wg**2 * np.sin(x)  # relative displacement
    steadystate_motions[1] = - amplitude_factor / wg    * np.cos(x)  # relative velocity
    steadystate_motions[2] =   amplitude_factor         * np.sin(x)  # relative acceleration
    
    return ground_acceleration, steadystate_motions
#########################################################################


#########################################################################
def sinusoidal_driven_oscillator(ts, Tg=1, ground_phase=0, natural_period=1, damping_ratio=0.1):
    """
    \t\n
    Return sinusoiudal ground acceleration of given ground period and phase, 
    the steady state relative motion and the transient response (displacement, velocity and acceleration)of 
    an harmonic oscillator evaluated at the times ts.
    """

    # setup 
    wg = 2*np.pi / Tg
    amplitude_factor, phase = setup_harmonic_oscillator(wg, natural_period, damping_ratio)
    
    # initial conditions
    U = amplitude_factor/wg**2 * np.sin(ground_phase+phase)
    V = amplitude_factor/wg    * np.cos(ground_phase+phase)
    
    # the homogeneous (or transient) solution
    transient_motions = damped_harmonic_oscillator(ts, U, V, natural_period, damping_ratio)
    transient_motions[:,ts<0] = 0
    
    # the particular solution
    ground_acceleration, steadystate_motions = steadystate_driven_oscillator(ts, Tg, ground_phase, natural_period, damping_ratio)
    ground_acceleration[ts<0] = 0
    steadystate_motions[:,ts<0] = 0

    # implementing eq. ()
    relative_motions = steadystate_motions + transient_motions
    
    return ground_acceleration, relative_motions, transient_motions
#########################################################################


#########################################################################
def test_driven_oscillator(ts, Tg=1, natural_period=1, damping_ratio=0.1):
    """
    \t\n
    Return the relative motion and the transient response (displacement, velocity and acceleration) of 
    an harmonic oscillator evaluated at the times ts driven by the test ground motion.
    """
    
    # setup 
    wg  = 2*np.pi/Tg
    wg2 = 2*wg
    
    A,  beta  = setup_harmonic_oscillator(wg,  natural_period, damping_ratio)
    A2, beta2 = setup_harmonic_oscillator(wg2, natural_period, damping_ratio)

    # initial conditions
    U  =   A  / wg**2  * np.cos(beta)
    V  = - A  / wg     * np.sin(beta)
    U -=   A2 / wg2**2 * np.cos(beta2)
    V -= - A2 / wg2    * np.sin(beta2)
    
    # steadystate motions
    x  = wg  * ts + beta
    x2 = wg2 * ts + beta2

    steadystate_motions = np.empty((3,)+ts.shape)
    steadystate_motions[0] = ( - A*np.cos(x) + A2*np.cos(x2)/4 ) / wg**2
    steadystate_motions[1] = (   A*np.sin(x) - A2*np.sin(x2)/2 ) / wg
    steadystate_motions[2] =     A*np.cos(x) - A2*np.cos(x2)
    
    steadystate_motions[:,ts<0]  = 0
    steadystate_motions[:,ts>=Tg]  = 0

    # transient motions
    transient_motions  = damped_harmonic_oscillator(ts, U, V, natural_period, damping_ratio)
    transient_motions2 = damped_harmonic_oscillator(ts-Tg, U, V, natural_period, damping_ratio)
    transient_motions[:,ts<0] = 0
    transient_motions2[:,ts<Tg] = 0
    transient_motions -= transient_motions2
    
    # relative motion
    relative_motions = steadystate_motions + transient_motions

    return relative_motions, transient_motions
#########################################################################


#########################################################################
def eva_FT_green(f, natural_period=1, damping_ratio=0.1):
    """
    \t\n
    Return the green function in the frequency domain evaluated 
    at the frequency (or frequencies) `f` for the harmonic oscillator 
    with given natural period and damping ratio
    """
    
    w0 = 2*np.pi/natural_period
    w = 2*np.pi*f
    FT_green = -1/(w0**2-w**2+2j*w*w0*damping_ratio)
    
    return FT_green
#########################################################################


#########################################################################
def eva_FT_response(f, natural_period=1, damping_ratio=0.1):
    """
    \t\n
    Return the response function in the frequency domain evaluated 
    at the frequency (or frequencies) `f` for the harmonic oscillator 
    with given natural period and damping ratio
    """
    
    w0 = 2*np.pi/natural_period
    w = 2*np.pi*f
    FT_response = w**2/(w0**2-w**2+2j*w*w0*damping_ratio)
    
    return FT_response
#########################################################################



# ## Seislab
# 
# Below we just copy the content of the file {Download}`seiswlab.py<./seislab.py>`

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

import cartopy


from scipy import optimize, linalg, stats, signal
import obspy


from obspy import Stream
from obspy.signal.util import _npts2nfft



###########################################
def setup_map(circle=None, extent=None, color=None, label=None, scale=1.1):
    """
    \t\n
    Return a Figure object with a geoaxes object from cartopy as axis.
    
    Parameters:
    -----------
    circle:   dictionary with keywords "longitude" and "latitude" used as 
              the centre of the map and "maxradius" as the radius [in degrees] 
              of the circle that must be contained in it.
    scale:    float that is used to scale the map extent [1.1 by default].
    extent:   array-like with min and max longitudes and min and max latitudes
              that is used to set the extent instead of the one derived from circle.
              If neither circle or extent are provided, the extent will be global.
    color:    color used to draw the circle and its centre [by default the circle is not drawn]
    label:    used for annotate the circle centre
    
    Return:
    -------
    fig:      the Figure object
        
    """

    if circle: 
        rad = circle["maxradius"]
        lon = circle["longitude"]
        lat = circle["latitude"]
        rad_m = rad * np.pi/180 * 6371009

    if extent:
        lon0 = np.mean(extent[:2])
        lat0 = np.mean(extent[2:])
    elif circle:
        lon0 = lon
        lat0 = lat
    else: 
        lon0,lat0 = None, None

    if lon0 is None: 
        projection = cartopy.crs.PlateCarree()
    else:
        projection = cartopy.crs.AzimuthalEquidistant(central_longitude=lon0, central_latitude=lat0)
        
    fig,ax = plt.subplots(tight_layout=True, figsize=(8,8), subplot_kw=dict(projection=projection))

    if extent is not None:
        ax.set_extent(extent)
    elif circle is not None:   
        extent = scale*np.array([-rad_m,rad_m,-rad_m,rad_m])
        ax.set_extent(extent,projection)
    else:
        ax.set_global()
        
    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS,color="gray");

    gl = ax.gridlines(draw_labels=True)
    gl.top_labels, gl.left_labels = False, False
    
    if circle and color:   
        rad_km = rad * np.pi/180 * 6371.009
        ax.tissot(rad_m/1e3, lon, lat, alpha=0.2, linewidth=2, edgecolor=color, facecolor="none")
        ax.scatter(lon,lat, transform=cartopy.crs.PlateCarree(), marker="*", color=color, zorder=30)
    if label:
        marker = ax.scatter(lon,lat, transform=cartopy.crs.PlateCarree(), marker="*", color=color, zorder=30)
        ax.annotate(label,(lon,lat-scale*rad/30), transform=cartopy.crs.PlateCarree(), color=marker._facecolors[0], ha='center',va="top")
         
    return fig
###########################################################



#################################################
def preprocessing(original_stream,order=1,taper=0.05):
    stream = original_stream.copy()
    stream.detrend(type="polynomial",order=order)
    stream.taper(taper,type="cosine")
    for trace in stream:
        n, d = trace.stats.npts, trace.stats.delta
        nn = _npts2nfft(n)
        T = (nn-n)*d
        endtime = trace.stats.endtime + T
        trace.trim(endtime=endtime,pad=True,fill_value=0)
    return stream
#################################################


###########################################################
def get_inverse_response(trace,output="DEF",last_stage=False):
    
    response = trace.stats.response
    end_stage = len(response.response_stages)
    if not last_stage: end_stage -= 1
    
    n, d = trace.stats.npts, trace.stats.delta    
    Rs, fs = response.get_evalresp_response(d,n,output=output,end_stage=end_stage)
    Rs[0] = np.inf
    IRs = 1/Rs
    
    return fs,IRs
###########################################################



###########################################################
def remove_response(original_stream,output="DEF"):
    stream = original_stream.copy()
    for trace in stream:
        fs,IRs = get_inverse_response(trace,output=output)
        fs,Zs,n,d = get_fft_trace(trace)
        trace.data = np.fft.irfft(Zs*IRs,n) / d
    return stream
###########################################################



###########################################################
def filtering(original_stream,BA_filters):
    stream = original_stream.copy()
    for trace,BA_filter in zip(stream,BA_filters):
        fs,Zs,n,d = get_fft_trace(trace)
        ws = 2*np.pi*fs
        ws, Hs = signal.freqs(*BA_filter, ws)
        trace.data = np.fft.irfft(Zs*Hs,n) / d
    return stream 
###########################################################


###########################################################
def filtering_zerophase(original_stream,butters):
    stream = original_stream.copy()
    for trace,butter in zip(stream,butters):
        fs,Zs,n,d = get_fft_trace(trace)
        ws = 2*np.pi*fs
        ws, Hs = signal.freqs(*butter, ws)
        zs = np.fft.irfft(Zs*Hs,n)
        zs = np.flip(zs)
        Zs = np.fft.rfft(zs)
        zs = np.fft.irfft(Zs*Hs,n) / d
        zs = np.flip(zs)
        trace.data = zs
    return stream
###########################################################




#################################################
def derivative(stream,order=1):
    """
    \t\n
    Return the time derivative of given order of the Stream object. The time derivative is evaluated in the frequency domain
    
    Parameters:
    -----------
    \tstream:\t Stream object to be derived
    \torder:\t Order of the time derivative
    
    Return:
    -------
    \tder_stream:  Derived Stream object
    
    Examples:
    ---------
    >>> from obspy import read
    >>> stream = read()
    >>> der_stream = derivative(stream)
    """

    der_stream = stream.copy()
    for trace in der_stream:
        fs,Zs,n,d = get_fft_trace(trace)
        ss = 2j*np.pi*fs
        ss = ss**order
        trace.data = np.fft.irfft(Zs*ss,n) / d

    return der_stream
#################################################



###########################################################
def remove_mean(stream,taper=0.05):
    """
    \t\n
    It remove the mean of the first decimal percentage of the data from the whole data time series of all the trace of a Stream object. The Stream object is modified in place.
    
    Parameters:
    -----------
    \tstream:\t Stream object 
    \ttaper:\t  Decimal percentage on which calculate the mean (5% by default)
    
    Examples:
    ---------
    >>> from obspy import read
    >>> stream = read()
    >>> remove_mean(stream)
    """

    for trace in stream:
        k = int(taper*trace.stats.npts)
        y = trace.data.astype(float)
        mean = y[:k].mean()
        trace.data = y - mean
###########################################################




###########################################################
def get_fft_trace(trace):
    
    n, d = trace.stats.npts, trace.stats.delta
    fs = np.fft.rfftfreq(n,d)
    Zs = np.fft.rfft(trace.data) * d
    
    return fs,Zs,n,d
###########################################################


###########################################################
def get_end_stage(response, end=False):
    end_stage = len(response.response_stages)
    stage = response.response_stages[-1]
    gain = 1
    if not end:
        if stage.input_units == "COUNTS": 
            end_stage -= 1
            gain = stage.stage_gain
    return end_stage,gain
###########################################################


###########################################################
def get_fft_response(trace, output="DEF", end=False):
    
    response = trace.stats.response
    sensitivity = response.instrument_sensitivity.value

    end_stage,gain = get_end_stage(response, end=end)
    
    n, d = trace.stats.npts, trace.stats.delta    
    Rs, fs = response.get_evalresp_response(d, n, output=output, end_stage=end_stage)
    Rs *= gain
    Rs[0] = np.inf
    IRs = 1/Rs
    
    return fs,Rs,IRs,sensitivity
###########################################################


###########################################################
def remove_response(stream, output="DEF", end=False):
    new = stream.copy()
    for trace in new:
        fs,Rs,IRs,sensitivity = get_fft_response(trace, output=output, end=end)
        fs,Zs,n,d = get_fft_trace(trace)
        trace.data = np.fft.irfft(Zs*IRs, n) / d
    return new
###########################################################


###########################################################
def filtering(stream, BA_filters):
    filtered_stream = stream.copy()
    for trace,BA_filter in zip(filtered_stream, BA_filters):
        fs,Zs,n,d = get_fft_trace(trace)
        ws = 2*np.pi*fs
        ws, Hs = signal.freqs(*BA_filter, ws)
        trace.data = np.fft.irfft(Zs*Hs,n) / d
    return filtered_stream 
###########################################################


###########################################################
def filtering_zerophase(stream,BA_filters):
    filtered_stream = stream.copy()
    for trace,BA_filter in zip(filtered_stream,BA_filters):
        fs,Zs,n,d = get_fft_trace(trace)
        ws = 2*np.pi*fs
        ws, Hs = signal.freqs(*BA_filter, ws)
        zs = np.fft.irfft(Zs*Hs,n) / d
        zs = np.flip(zs)
        Zs = np.fft.rfft(zs) * d
        zs = np.fft.irfft(Zs*Hs,n) / d
        zs = np.flip(zs)
        trace.data = zs
    return filtered_stream
###########################################################

###########################################################
def get_channels_with_orientation(iterable):
    channels = []
    for elem in iterable:
        if type(elem) == obspy.core.inventory.channel.Channel:
            channel = elem.code
        else:
            channel = elem.stats.channel
        if channel not in channels: channels.append(channel)
    return channels
###########################################################


###########################################################
def get_channels(iterable, return_indices=False):
    channels = []
    indices = []
    for k, elem in enumerate(iterable):
        if type(elem) == obspy.core.inventory.channel.Channel:
            channel = elem.code[:2]+"*"
        else:
            channel = elem.stats.channel[:2]+"*"
        if channel not in channels: 
            channels.append(channel)
            indices.append(k)
    if return_indices:
        return (channels, indices)
    else:
        return channels
###########################################################





###########################################################
def make_label(trace_id,label_id,extra_label):

    label = None
    
    if label_id:
        label = trace_id
        if extra_label is not None: 
            if extra_label[:2] == "\n":
                label += extra_label
            else:
                label += " "+extra_label
    else:
        if extra_label is not None: label = extra_label
    
    return label
###########################################################



"""
###########################################################
def make_label(trace,label_id,extra_label):

    label = None
    
    if label_id:
        label = trace.stats.channel
        if extra_label is not None: label += extra_label
    else:
        if extra_label is not None: label = extra_label
    
    return label
###########################################################
"""

###########################################################
def get_gaps(stream,reftime):
    
    gaps = stream.get_gaps()
    channels = get_channels_with_orientation(stream)
    dict_gaps = {}
    for channel in channels:
        dict_gaps[channel] = np.array([ [gap[4]-reftime,gap[5]-reftime] for gap in gaps if gap[3] == channel])
   
    return dict_gaps
###########################################################



###########################################################
def plot_time(stream, reftime=None, picks=None,
                      remove_sensitivity=False, fig=None,
                      ncol=1, col=0, sharey=False,
                      label_id=True, extra_label=None, taper=None, title=None, 
                      linewidth=1, color=None, alpha=1):
    
    logy_color = color is None

    if extra_label is not None: label_id = False
    
    logy_trace = type(stream) == obspy.core.trace.Trace
    if logy_trace: stream = obspy.Stream(stream)
    
    if reftime is None: reftime = min(trace.stats.starttime for trace in stream)

    channels = get_channels_with_orientation(stream)

    dict_gaps = get_gaps(stream,reftime)
    dict_picks = get_picks(stream,reftime,picks)
    
    logy = fig is None
    
    nrow = len(channels)

    
    if logy:
        height = 2*nrow+0.5
        if title is not None: height += 1
        fig,axes = plt.subplots(nrow,ncol,tight_layout=True,sharex="col",sharey=sharey,figsize=(10,height))
        if nrow == 1 and ncol == 1: axes = np.array([axes])
        if title is not None: fig.suptitle(title)
        axes = np.array(axes).reshape(nrow,-1)
        for ax in axes[-1]:
            ax.set_xlabel("Time [s] with respect to the origin time")
    else:
        axes = fig.axes
    axes = np.array(axes).reshape(nrow,-1)
    
    dict_axes = {}
    for ax,channel in zip(axes[:,col],channels):
        
        now = stream.select(channel=channel)    
                         
        for k,trace in enumerate(now):
    
            ts = trace.times() 
            ts += trace.stats.starttime - reftime

            if k == 0: label = make_label(trace.stats.channel,label_id,extra_label)
            else: label = None

            zs = trace.data.copy()
            if remove_sensitivity: zs = zs.astype(float) / trace.stats.response.instrument_sensitivity.value
            
            line, = ax.plot(ts,zs, label=label, color=color, linewidth=linewidth, alpha=alpha)
            if logy_color: color = line._color

            if label is not None: ax.legend()

            if trace.data.max() > 0 and trace.data.min() < 0:
                ax.axhline(0, color="gray", linewidth=0.5)
                
            if taper is not None:
                k = int(trace.stats.npts*0.05)
                T = trace.stats.delta * k
                ax.axvspan(ts[0],    ts[0]+T, color="yellow", alpha=0.2)
                ax.axvspan(ts[-1]-T, ts[-1],  color="yellow", alpha=0.2)

        for t0,t1 in dict_gaps[channel]:
            ax.axvspan(t0,t1, color="red", alpha=0.2)
        for t0,dt in dict_picks[channel]:
            ax.axvline(t0, color="green")
            ax.axvspan(t0-dt,t0+dt, color="green", alpha=0.2)
    
    if logy_trace: stream = stream[0]
    
    return fig
###########################################################




###########################################################
def get_gaps(stream,reftime):
    
    gaps = stream.get_gaps()
    channels = get_channels_with_orientation(stream)
    dict_gaps = {}
    for channel in channels:
        dict_gaps[channel] = np.array([ [gap[4]-reftime,gap[5]-reftime] for gap in gaps if gap[3] == channel])
   
    return dict_gaps
###########################################################



###########################################################
def get_picks(stream,reftime,picks):
    
    channels = get_channels_with_orientation(stream)
    dict_picks = {}
    for channel in channels:
        dict_picks[channel] = []
        if picks is None: continue
        now = stream.select(channel=channel)
        trace = now[0]
        for pick in picks:
            if trace.id == pick.waveform_id.id:
                dict_picks[channel].append([pick.time-reftime,pick.time_errors])
   
    return dict_picks
###########################################################


###########################################################
def get_channels(iterable, return_indices=False):
    channels = []
    indices = []
    for k, elem in enumerate(iterable):
        if type(elem) == obspy.core.inventory.channel.Channel:
            channel = elem.code[:2]+"*"
        else:
            channel = elem.stats.channel[:2]+"*"
        if channel not in channels: 
            channels.append(channel)
            indices.append(k)
    if return_indices:
        return (channels, indices)
    else:
        return channels
###########################################################





###########################################################
def make_label(trace_id,label_id,extra_label):

    label = None
    
    if label_id:
        label = trace_id
        if extra_label is not None: 
            if extra_label[:2] == "\n":
                label += extra_label
            else:
                label += " "+extra_label
    else:
        if extra_label is not None: label = extra_label
    
    return label
###########################################################



    
###########################################################
def plot_fft(stream, 
             fig=None, nrow=1, row=0, sharey="row", xscale="log",
             label_id=True, extra_label=None, title=None,
             bands=None, remove_sensitivity=False, 
             linewidth=1, color=None, alpha=1):

    if extra_label is not None: label_id = False


    logy_trace = type(stream) == obspy.core.trace.Trace
    if logy_trace: stream = obspy.Stream(stream)

    logy = fig is None
    
    if bands is None: bands = [None for _ in stream]
    
    ncol = len(stream)
    if ncol == 6: 
        ncol //= 2
        nrow *= 2
    
    if logy:
        height = 3*nrow+0.5
        if title is not None: height += 1
        fig,axes = plt.subplots(nrow, ncol, tight_layout=True, sharex=True, sharey=sharey, figsize=(10,height))
        if nrow == 1 and ncol == 1: axes = np.array([axes])
        axes = axes.reshape((nrow,-1))
        if title is not None: fig.suptitle(title)
        for ax in axes.flatten():
            ax.set_xscale(xscale)
        for ax in axes[-1]:
            ax.set_xlabel("Frequency [hz]")
    else:
        axes = fig.axes
    axes = np.array(axes).reshape(-1,len(stream))

    for ax, trace, band in zip(axes[row], stream, bands):
    
        fs,Zs,n,d = get_fft_trace(trace)
        if remove_sensitivity: Zs /= trace.stats.response.instrument_sensitivity.value
            
        label = make_label(trace.stats.channel,label_id,extra_label)
        
        plot_spectrum(fs, Zs, ax, label=label, band=band, linewidth=linewidth, color=color, alpha=alpha)
                   
    if logy_trace: stream = stream[0]


    return fig
###########################################################


###########################################################
def plot_spectrum(fs, Rs, axes, label=None, band=None, linewidth=1, color=None, alpha=1, hline=None):
    
    logy = type(axes) == np.ndarray or type(axes) == list
        
    if logy:
        ax = axes[0]
        axphase = axes[1]
    else:
        ax = axes
    
    amplitude = abs(Rs)
    ax.plot(fs[1:],amplitude[1:],label=label, linewidth=linewidth, color=color, alpha=alpha)
    ax.set_yscale("log")

    if logy:
        angles = np.angle(Rs,deg=True)
        axphase.plot(fs[1:], angles[1:], label=label, linewidth=linewidth, color=color, alpha=alpha)
        axphase.set_yticks([-180,-90,0,90,180])
        axphase.set_yscale("linear")
        for val in [-180,0,180]:
            axphase.axhline(val,color="gray",linewidth=0.5)
            
    if band is not None:
        if len(band) > 1:
            ax.axvspan(*band, linewidth=0.5, color="red", alpha=0.2)
            if logy: axphase.axvspan(*band, linewidth=0.5, color="red", alpha=0.2)
        else:
            ax.axvline(band, linewidth=0.5, color="red")
            if logy: axphase.axvline(band, linewidth=0.5, color="red")
    
    if hline: ax.axhline(hline,color="red",linewidth=0.5)
    
    if label is not None: 
        ax.legend()
        if logy: axphase.legend()
###########################################################
            


###########################################################
def plot_response(stream, output="DEF", fig=None, title=None, extra_label=None, bands=None, sharey=False, xscale="log", linewidth=1, color=None, alpha=1):

    logy_trace = type(stream) == obspy.core.trace.Trace
    if logy_trace: stream = obspy.Stream(stream)

    logy = fig is None
    

    channels, indices = get_channels(stream, return_indices=True)

    if bands is None: bands = np.array([None for _ in stream])
    newbands = bands[indices]

    if logy:
        height = 6 + 0.5
        if title is not None: height += 1
        fig,axes = plt.subplots(2,len(channels),tight_layout=True,sharex=True,sharey=sharey,figsize=(10,height))
        if title is not None: fig.suptitle(title)
        for ax in axes.flatten():
            ax.set_xscale(xscale)
        axes = axes.reshape((2,-1))
        for ax in axes[-1]:
            ax.set_xlabel("Frequency [hz]")
        axes[0,0].set_ylabel("Amplitude")
        axes[1,0].set_ylabel("Phase [deg]")
    else:
        axes = np.resphape(fig.axes,(2,-1))
    
    for axe,channel,band in zip(axes.T,channels,newbands):
        
        trace = stream.select(channel=channel)[0]
    
        fs,Rs,IRs,sensitivity = get_fft_response(trace,output=output,endstage=endstage)
        
        label = make_label(channel, logy, extra_label)
        
        plot_spectrum(fs, Rs, axes=axe, label=label, band=band,
                      linewidth=linewidth, color=color, alpha=alpha)

    if logy_trace: stream = stream[0]
    return fig
###########################################################






###########################################################
def keep_seismic_channels(inventory):

    band_codes = ["F","G","D","C","E","H","B","M","L","V","U","R","P","T","Q","A","O"]
    inst_codes = ["N","L","H"]

    for network in inventory:
        for k,station in enumerate(network):
            codes = np.unique([ channel.code[:2]+"*" for channel in station ])
            channels = []
            for inst in inst_codes:
                bands = np.unique([ code[0] for code in codes if code[1] == inst ])
                channel = None
                for selected_band in band_codes:
                    if selected_band in bands:
                        channel = [band+inst+"*" for band in bands if band != selected_band]
                        break
                if channel is not None:
                    channels += channel
            for channel in channels:
                inventory = inventory.remove(network.code,station.code,"*",channel)
                
    return inventory
###########################################################













###########################################################s
def setup_extent(circle):
    
    dlat, lon, lat = circle.values()
    dlat *= 1.1
    dlon = dlat / np.cos(lat*np.pi/180)
    
    extent = [lon-dlon, lon+dlon, lat-dlat, lat+dlat]
    
    return extent
###########################################################



###########################################################
def setup_geofig(circle=None, extent=None, color=None, label=None, fig=None):
    
    if extent:
        lon = np.mean(extent[:2])
        lat = np.mean(extent[2:])
    elif circle:
        lon = circle["longitude"]
        lat = circle["latitude"]
    else: 
        lon,lat = None, None

    if lon is None: 
        projection = cartopy.crs.PlateCarree()
    else:
        projection = cartopy.crs.AzimuthalEquidistant(central_longitude=lon, central_latitude=lat)

    if fig is None:
        fig,ax = plt.subplots(tight_layout=True, figsize=(8,8), subplot_kw=dict(projection=projection))
    else:
        ax = fig.axes[0]

    if circle and not extent: extent = setup_extent(circle)    
    if extent is not None:    ax.set_extent(extent)
    
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels, gl.left_labels = False, False
    ax.coastlines(linewidth=0.5)
    
    if circle and color:   
        rad_deg, lon, lat = circle.values() 
        rad_km = rad_deg * np.pi/180 * 6371.009
        ax.tissot(rad_km, lon, lat, alpha=0.2, linewidth=2, edgecolor=color, facecolor="none")
        ax.scatter(lon,lat, transform=cartopy.crs.PlateCarree(), marker="*", color=color)
    if label:
        dlat = (extent[3]-extent[2])/50
        marker = ax.scatter(lon,lat, transform=cartopy.crs.PlateCarree(), marker="*", color=color)
        ax.annotate(label,(lon,lat-dlat), transform=cartopy.crs.PlateCarree(), color=marker._facecolors[0], ha='center',va="top")
         
    return fig
###########################################################






###########################################################
def setup_geofigs(nc,nr=1,circle=None, extent=None, circle_color=None, fig=None):
    
    if extent:
        lon = np.mean(extent[:2])
        lat = np.mean(extent[2:])
    elif circle:
        _, lon, lat = circle.values() 

    projection = cartopy.crs.AzimuthalEquidistant(central_longitude=lon, central_latitude=lat)

    if fig is None:
        size = (10-0.5)/nc
        fig,axes = plt.subplots(nr,nc,tight_layout=True, figsize=(size*nc+0.5,size*nr+0.5), subplot_kw=dict(projection=projection))
    else:
        axes = fig.axes
    axes = axes.flatten()
    
    if circle and not extent: 
        extent = setup_extent(circle)  

    for ax in axes:
        ax.set_extent(extent)
        
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels, gl.left_labels = False, False
        ax.coastlines(linewidth=0.5)
        
        if circle and circle_color:   
            rad_deg, lon, lat = circle.values() 
            rad_km = rad_deg * np.pi/180 * 6371.009
            ax.tissot(rad_km, lon, lat, alpha=0.2, linewidth=2, edgecolor=circle_color, facecolor="none")
            ax.scatter(lon,lat, transform=cartopy.crs.PlateCarree(), marker="*", color=circle_color)
         
    return fig
###########################################################



###########################################################s
def plot_data(data,origin=None,modelled_times=None,fig=None):
    
    if origin is None:
        origin = np.zeros(4)
    
    T0,X0,Y0,Z0 = origin
    
    Ls = np.sqrt((data[1]-X0)**2+(data[2]-Y0)**2)
    Ds = np.sqrt(Ls**2+(data[3]-Z0)**2)
    Ts = data[0]
    Es = data[4]
    phases = data[-1]
    if modelled_times is not None:
        Rs = Ts - modelled_times
    else:
        Rs = np.zeros(len(Ts))
            
    D_max = 1.05*Ds.max()
    T_max = 1.05*Ts.max()
    R_max = 1.05*( abs(Rs).max() + abs(Es).max() )
    
    if not fig:
        fig,axes = plt.subplots(1,2,tight_layout=True,figsize=(12,4),sharey=True)
    else:
        axes = fig.axes
        
    for phase in range(2):
        whe = phases == phase
        if phase == 1: label = "S data"
        elif phase == 0: label = "P data"
        axes[0].errorbar(Ts[whe],Ds[whe],xerr=Es[whe],fmt=".",linewidth=0.5,label=label)
        axes[1].errorbar(Rs[whe],Ds[whe],xerr=Es[whe],fmt=".",linewidth=0.5)
    if modelled_times is not None:
        axes[0].scatter(modelled_times,Ds,marker="*",color="red",label="model")
    axes[0].legend()
    axes[0].set_ylabel("Hypocentral distance [km]")    
    axes[0].set_xlabel("Times [s]")    
    axes[1].set_xlabel("Residues [s]")    
    
    for k,ax in enumerate(axes):
        ax.set_ylim(0,D_max)
        if k == 0:
            ax.axvline(T0,linewidth=0.5,color="black")
            ax.text(T0,0.9*D_max,"Origin Time",rotation="vertical",horizontalalignment="center",verticalalignment="top",backgroundcolor="white")
        else:
            ax.set_xlim(-R_max,R_max)
            ax.axvline(0,linewidth=0.5,color="black")
        
    return fig
###########################################################s



###########################################################s
def plot_conditional_probability(keys,qb,qb_std,functional,nsigma=4,qb_std_test=None):

    M1 = len(qb)
    nr = int(np.ceil(M1/2))
    fig,axes = plt.subplots(nr,2,figsize=(12,8),tight_layout=True)
    axes = axes.flatten()

    for k,key in enumerate(keys):
        ax = axes[k]
        ax.set_xlabel(key)
        vals = np.linspace(-nsigma,nsigma,1000)*qb_std[k] + qb[k]
        if k == 3 and vals[-1] >= 0: 
            vals = vals.clip(-np.inf,0)
            ax.axvline(0,color="black",linewidth=0.5)

        Ls = np.empty(len(vals))
        q = qb.copy()
        for i,val in enumerate(vals):
            q[k] = val
            Ls[i] = functional(q)

        Ls -= Ls.min()
        prob = np.exp(-Ls)
        norma = ( (prob[1:]+prob[:-1])*np.diff(vals)/2 ).sum()
        prob /= norma
        ax.plot(vals,prob,label="conditional\nprobability")
        
        gaus = stats.norm(qb[k],qb_std[k])
        ax.plot(vals,gaus.pdf(vals),label="gaussian")

        if qb_std_test is not None:
            gaus = stats.norm(qb[k],qb_std_test[k])
            ax.plot(vals,gaus.pdf(vals),label="gaussian 2",color="red",linestyle="dashed")

        ax.axhline(0,color="black",linewidth=0.5)
        ax.axvline(qb[k],color="red")
        ax.axvspan(qb[k]-qb_std[k],qb[k]+qb_std[k],alpha=0.2,color="red")
        ax.legend()

    for ax in axes[M1:]:
        ax.set_visible(False)

    return fig
###########################################################s



###########################################################s
def unpack_model(model):
    """
    ...
    return (VP,VS,W) \n
    Return the P and S wave velocities `VP` and `VS`, and the thicknesses `W` of the layers of the velocity `model`
    """
    
    n = (len(model)+1)//3
    VP = np.array(model[:n])
    VS = np.array(model[n:2*n])
    W = np.array(model[2*n:])    
    
    return (VP,VS,W)
###########################################################s



###########################################################s
def setup_hypocenter_depth(Z0,model):
    """
    ...
    return (VP,VS,W,i0) \n
    Setup the model including the hypocenter depth `Z0` within the velocity model, assuming that the seismic station has no elevation.
    Return the P and S wave velocities `VP` and `VS`, the thicknesses `W` of the layers and the index `i0` of the new layer at the hypocenter depth.
    """
    #print("\n setup",Z0)

    VP,VS,W = unpack_model(model)
    H = np.append(0,np.cumsum(W))

    if Z0 not in H:
        i0 = np.where(H < Z0)[0][-1] + 1
        H = np.concatenate((H[:i0],[Z0],H[i0:]))
        VP = np.concatenate((VP[:i0],[VP[i0-1]],VP[i0:]))
        VS = np.concatenate((VS[:i0],[VS[i0-1]],VS[i0:]))
    else:
        i0 = np.where(H == Z0)[0][0] 

    W = np.diff(H)
        
    return (VP,VS,W,i0)
###########################################################s



###########################################################s
def setup_elevation(Z,V,W,i0):
    """
    ...
    return (WW,VV,j0) \n
    Return the modified the (P or S) wave velocities `VV`, thicknesses `WW` of the layer and the index `j0` of the new layer at the hypocenter depth in order to account for the elevation `Z` of the seismic station. The elevation must be negative (above the sea level).
    """

    if Z == 0:
        
        return (W,V,i0)
    
    elif Z < 0:
        
        if i0 == 0:
            WW = np.append(-Z,W)
            VV = np.append(V[0],V)
            j0 = i0 + 1
            return (WW,VV,j0)
        else:
            WW = W.copy()
            WW[0] -= Z
            return (WW,V,i0)
        
    else:
        
        print("Warning! The elevation must be negative (above the sea level)")

        
        
def eva_first_arrival(L,Z,Z0,V,W,i0):
    """
    ...
    return T \n
    Return the travel time `T` of the first arrival given the epicentral distance [L], the elevation [Z] of the seismic station, the hypocenter depth [Z0], and the (P or S) wave velocities and the thicknesses [W] as returned by the functions `setup_first_arrival` and `setup_elevation`
    """
    
    T_min = np.inf

    WW,VV,i0 = setup_elevation(Z,V,W,i0)
    WdV = WW/VV[:-1]
    WbV = WW*VV[:-1]

    ### HEAD WAVE #########################################
    n = len(VV)
    
    if VV[i0-1] == VV[i0]: j = i0+1
    else: j = i0
    
    for i in range(j,n):
        p = 1/VV[i]
        SS = VV[:i]*p 

        if (SS < 1).all():
            
            CC = np.sqrt(1-SS**2)
            LL = 2*(WbV[i0:i]/CC[i0:]).sum() + (WbV[:i0]/CC[:i0]).sum()
            if L < LL*p: continue

            T = L*p + 2*(WdV[i0:i]*CC[i0:]).sum() + (WdV[:i0]*CC[:i0]).sum()
            T_min = min(T_min,T)
    #######################################################

    ### DIRECT WAVE ########################################
    if i0 == 0:
        
        T = L/VV[0]
        
    elif i0 == 1:
        
        D = np.sqrt(L**2 + WW[0]**2)
        T = D/VV[0]

    else:
        
        VV = VV[:i0]
        WW = WW[:i0]
        WdV = WdV[:i0]
        WbV = WbV[:i0]

        imax = np.argmax(VV)

        D = np.sqrt(L**2+(Z0-Z)**2)
        p0 = L/D/VV[imax]
        
        D = np.sqrt(L**2+WW[imax]**2)
        p1 = L/D/VV[imax]

        fun = lambda p: L - p * ( WbV/np.sqrt(1-(VV*p)**2) ).sum()

        if fun(p0)*fun(p1) < 0: sol = optimize.brentq(fun,p0,p1)
        else: sol = p0

        T = L*sol + ( WdV*np.sqrt(1-(VV*sol)**2) ).sum()
    #######################################################

    T_min = min(T_min,T) 
    
    return T_min
############################################################


############################################################
def eva_first_arrivals(Ls,Zs,Z0,model,phase=0):
    """
    ...
    return Ts \n
    Return the travel times `Ts` of the first (P or S wave) arrivals given the epicentral distances [Ls], the elevations [Zs] of the seismic stations, the hypocenter depth [Z0], and velocity model `model` 
    """
    
    if type(Zs) is not np.ndarray:
        Zs = Zs*np.ones(len(Ls))

    VP,VS,W,i0 = setup_hypocenter_depth(Z0,model)
    Ts = np.empty(len(Ls))
    for i,(L,Z) in enumerate(zip(Ls,Zs)):
        if phase: T = eva_first_arrival(L,Z,Z0,VS,W,i0)
        else:     T = eva_first_arrival(L,Z,Z0,VP,W,i0)
        Ts[i] = T
    return Ts
###########################################################s


# In[ ]:




